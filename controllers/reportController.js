import Report from '../models/Report.js';
import Patient from '../models/Patient.js';
import Trial from '../models/Trial.js';
import { logger } from '../utils/logger.js';
import { 
  storePatientReport, 
  storeFinalReport 
} from '../services/blockchainService.js';
import { 
  analyzeReport, 
  mockAnalyzeReport 
} from '../services/aiService.js';

/**
 * @desc    Submit a patient report
 * @route   POST /api/reports/submit
 * @access  Private (Hospital only)
 */
export const submitPatientReport = async (req, res) => {
  try {
    const {
      patientId,
      age,
      gender,
      trialId,
      trialStartDate,
      trialEndDate,
      drugName,
      dosage,
      preExistingConditions,
      observedSideEffects,
      sideEffectSeverity,
      symptomImprovementScore,
      overallHealthStatus,
      doctorNotes
    } = req.body;

    // Validate required fields
    if (!patientId || !trialId || !trialStartDate || !trialEndDate || !drugName || !dosage) {
      return res.status(400).json({
        success: false,
        error: 'Please provide all required fields'
      });
    }

    // Check if trial exists and hospital is part of it
    const trial = await Trial.findById(trialId);
    
    if (!trial) {
      return res.status(404).json({
        success: false,
        error: 'Trial not found'
      });
    }

    // Check if hospital is part of this trial and has accepted
    const hospitalInTrial = trial.hospitals.find(
      h => h.hospital.toString() === req.user.id && h.status === 'accepted'
    );

    if (!hospitalInTrial) {
      return res.status(403).json({
        success: false,
        error: 'Hospital not authorized for this trial'
      });
    }

    // Check if patient exists, create if not
    let patient = await Patient.findOne({ patientId });
    
    if (!patient) {
      patient = await Patient.create({
        patientId,
        age,
        gender,
        hospital: req.user.id,
        trial: trialId
      });
    } else {
      // Verify patient belongs to this hospital and trial
      if (patient.hospital.toString() !== req.user.id || patient.trial.toString() !== trialId) {
        return res.status(403).json({
          success: false,
          error: 'Patient does not belong to this hospital or trial'
        });
      }
    }

    // Check if report for this patient already exists
    const existingReport = await Report.findOne({
      patient: patient._id,
      trial: trialId
    });

    if (existingReport) {
      return res.status(400).json({
        success: false,
        error: 'Report for this patient already exists'
      });
    }

    // Create report object
    const reportData = {
      patient: patient._id,
      trial: trialId,
      hospital: req.user.id,
      trialStartDate,
      trialEndDate,
      drugName,
      dosage,
      preExistingConditions: preExistingConditions || [],
      observedSideEffects: observedSideEffects || [],
      sideEffectSeverity,
      symptomImprovementScore,
      overallHealthStatus,
      doctorNotes
    };

    // Create report in database (without hash initially)
    const report = await Report.create(reportData);

    // Store hash on blockchain
    const blockchainResult = await storePatientReport(report);

    // Update report with hash
    report.reportHash = blockchainResult.dataHash;
    await report.save();

    logger.info(`New patient report submitted: ${report._id} for patient ${patientId}`);

    res.status(201).json({
      success: true,
      data: report,
      blockchain: {
        transactionHash: blockchainResult.transactionHash,
        blockNumber: blockchainResult.blockNumber
      }
    });
  } catch (error) {
    logger.error(`Report submission error: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

/**
 * @desc    Get reports for a trial
 * @route   GET /api/reports/trial/:trialId
 * @access  Private (Hospital, Manufacturer, Regulator)
 */
export const getTrialReports = async (req, res) => {
  try {
    const { trialId } = req.params;

    // Check if trial exists
    const trial = await Trial.findById(trialId);
    
    if (!trial) {
      return res.status(404).json({
        success: false,
        error: 'Trial not found'
      });
    }

    // Check if user has access to this trial
    const hasAccess = 
      req.user.role === 'regulator' ||
      trial.manufacturer.toString() === req.user.id ||
      trial.hospitals.some(h => h.hospital.toString() === req.user.id);

    if (!hasAccess) {
      return res.status(403).json({
        success: false,
        error: 'Not authorized to access reports for this trial'
      });
    }

    // Get reports based on user role
    let query = { trial: trialId };
    
    // Hospitals can only see their own reports
    if (req.user.role === 'hospital') {
      query.hospital = req.user.id;
    }

    const reports = await Report.find(query)
      .populate('patient', 'patientId age gender')
      .populate('hospital', 'name organization');

    res.status(200).json({
      success: true,
      count: reports.length,
      data: reports
    });
  } catch (error) {
    logger.error(`Get trial reports error: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

/**
 * @desc    Finalize trial reports and generate AI analysis
 * @route   POST /api/reports/finalize/:trialId
 * @access  Private (Hospital only)
 */
export const finalizeTrialReports = async (req, res) => {
  try {
    const { trialId } = req.params;

    // Check if trial exists
    const trial = await Trial.findById(trialId);
    
    if (!trial) {
      return res.status(404).json({
        success: false,
        error: 'Trial not found'
      });
    }

    // Check if hospital is part of this trial and has accepted
    const hospitalInTrial = trial.hospitals.find(
      h => h.hospital.toString() === req.user.id && h.status === 'accepted'
    );

    if (!hospitalInTrial) {
      return res.status(403).json({
        success: false,
        error: 'Hospital not authorized for this trial'
      });
    }

    // Get all reports for this hospital and trial
    const reports = await Report.find({
      trial: trialId,
      hospital: req.user.id
    }).populate('patient', 'patientId age gender');

    console.log(JSON.stringify(reports, null, 2));


    if (reports.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'No reports found for this trial'
      });
    }

    // Check if final report already exists
    const existingFinalReport = await Report.findOne({
      trial: trialId,
      hospital: req.user.id,
      isFinal: true
    });

    if (existingFinalReport) {
      return res.status(400).json({
        success: false,
        error: 'Final report already exists for this trial'
      });
    }

    // Aggregate report data
    const totalPatients = reports.length;
    const averageImprovementScore = reports.reduce((sum, report) => sum + report.symptomImprovementScore, 0) / totalPatients;
    
    // Calculate side effect distribution
    const sideEffectDistribution = {
      None: 0,
      Mild: 0,
      Moderate: 0,
      Severe: 0
    };
    
    reports.forEach(report => {
      sideEffectDistribution[report.sideEffectSeverity]++;
    });
    
    // Calculate health status distribution
    const healthStatusDistribution = {
      Worse: 0,
      Same: 0,
      Improved: 0
    };
    
    reports.forEach(report => {
      healthStatusDistribution[report.overallHealthStatus]++;
    });
    
    // Find common side effects
    const allSideEffects = reports.flatMap(report => report.observedSideEffects);
    const sideEffectCounts = {};
    
    allSideEffects.forEach(effect => {
      sideEffectCounts[effect] = (sideEffectCounts[effect] || 0) + 1;
    });
    
    const commonSideEffects = Object.entries(sideEffectCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([effect, count]) => ({ effect, count }));

    // Create final report data
    const finalReportData = {
      patient: reports[0].patient._id, // Use first patient as reference
      trial: trialId,
      hospital: req.user.id,
      trialStartDate: reports[0].trialStartDate,
      trialEndDate: reports[0].trialEndDate,
      drugName: reports[0].drugName,
      dosage: reports[0].dosage,
      sideEffectSeverity: 'None', // Placeholder
      symptomImprovementScore: averageImprovementScore,
      overallHealthStatus: 'Same', // Placeholder
      doctorNotes: `Final report for ${totalPatients} patients`,
      isFinal: true,
      // Additional aggregated data
      totalPatients,
      averageImprovementScore,
      sideEffectDistribution,
      healthStatusDistribution,
      commonSideEffects,
      patientReports: reports.map(report => report._id)
    };

    // Create final report in database (without hash initially)
    const finalReport = await Report.create(finalReportData);

    

    // Generate AI analysis
    let aiAnalysisResult;
    
    try {
      if (process.env.NODE_ENV === 'production' && process.env.AI_SERVICE_URL) {
        aiAnalysisResult = await analyzeReport(finalReport);
      } else {
        // Use mock analysis for development/testing
        aiAnalysisResult = mockAnalyzeReport(finalReport);
      }
      
      finalReport.aiAnalysis = aiAnalysisResult.analysis;
    } catch (error) {
      logger.error(`AI analysis error: ${error.message}`);
      // Continue even if AI analysis fails
      finalReport.aiAnalysis = {
        error: 'AI analysis failed',
        message: error.message
      };
    }

    // Store final report with AI analysis on blockchain
    const blockchainResult = await storeFinalReport(finalReport, finalReport.aiAnalysis);

    // Update final report with hashes
    finalReport.reportHash = blockchainResult.dataHash;
    finalReport.aiAnalysisHash = blockchainResult.dataHash; // Same hash for simplicity
    await finalReport.save();

    // Update trial status if all hospitals have submitted final reports
    const allHospitals = trial.hospitals.filter(h => h.status === 'accepted');
    const finalReportsCount = await Report.countDocuments({
      trial: trialId,
      isFinal: true
    });

    if (finalReportsCount === allHospitals.length && trial.status === 'in_progress') {
      trial.status = 'completed';
      trial.completedAt = new Date();
      await trial.save();
    }

    logger.info(`Final report submitted for trial ${trialId} by hospital ${req.user.id}`);

    res.status(201).json({
      success: true,
      data: finalReport,
      blockchain: {
        transactionHash: blockchainResult.transactionHash,
        blockNumber: blockchainResult.blockNumber
      }
    });
  } catch (error) {
    logger.error(`Finalize reports error: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

/**
 * @desc    Get final reports for a trial
 * @route   GET /api/reports/final/:trialId
 * @access  Private (Manufacturer, Regulator)
 */
export const getFinalReports = async (req, res) => {
  try {
    const { trialId } = req.params;

    // Check if trial exists
    const trial = await Trial.findById(trialId);
    
    if (!trial) {
      return res.status(404).json({
        success: false,
        error: 'Trial not found'
      });
    }

    // Check if user has access to this trial
    const hasAccess = 
      req.user.role === 'regulator' ||
      trial.manufacturer.toString() === req.user.id;

    if (!hasAccess) {
      return res.status(403).json({
        success: false,
        error: 'Not authorized to access final reports for this trial'
      });
    }

    // Get final reports
    const finalReports = await Report.find({
      trial: trialId,
      isFinal: true
    }).populate('hospital', 'name organization');

    res.status(200).json({
      success: true,
      count: finalReports.length,
      data: finalReports
    });
  } catch (error) {
    logger.error(`Get final reports error: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

