import Trial from '../models/Trial.js';
import Report from '../models/Report.js';
import { logger } from '../utils/logger.js';
import { 
  storeRegulatorDecision,
  verifyHashOnBlockchain 
} from '../services/blockchainService.js';
import { generateHash } from '../utils/hashUtils.js';

/**
 * @desc    Verify and approve/reject a trial
 * @route   POST /api/regulator/verify/:trialId
 * @access  Private (Regulator only)
 */
export const verifyTrial = async (req, res) => {
  try {
    const { trialId } = req.params;
    const { decision, comments } = req.body;

    if (!decision || !['approved', 'rejected'].includes(decision)) {
      return res.status(400).json({
        success: false,
        error: 'Please provide a valid decision (approved/rejected)'
      });
    }

    // Check if trial exists and is completed
    const trial = await Trial.findById(trialId);
    
    if (!trial) {
      return res.status(404).json({
        success: false,
        error: 'Trial not found'
      });
    }

    if (trial.status !== 'completed') {
      return res.status(400).json({
        success: false,
        error: 'Trial is not yet completed'
      });
    }

    if (trial.regulatorDecision.decision !== 'pending') {
      return res.status(400).json({
        success: false,
        error: 'Trial has already been verified'
      });
    }

    // Store decision on blockchain
    const blockchainResult = await storeRegulatorDecision(
      trial,
      { _id: req.user.id },
      decision,
      comments || ''
    );

    // Update trial with regulator decision
    trial.regulatorDecision = {
      decision,
      regulator: req.user.id,
      decisionDate: new Date(),
      decisionHash: blockchainResult.dataHash,
      comments: comments || ''
    };

    trial.status = decision;
    await trial.save();

    logger.info(`Regulator ${req.user.id} ${decision} trial ${trialId}`);

    res.status(200).json({
      success: true,
      data: trial,
      blockchain: {
        transactionHash: blockchainResult.transactionHash,
        blockNumber: blockchainResult.blockNumber
      }
    });
  } catch (error) {
    logger.error(`Trial verification error: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

/**
 * @desc    Verify data integrity for a trial
 * @route   GET /api/audit/:trialId
 * @access  Private (Regulator only)
 */
export const auditTrial = async (req, res) => {
  try {
    const { trialId } = req.params;

    // Check if trial exists
    const trial = await Trial.findById(trialId)
      .populate('manufacturer', 'name organization')
      .populate('hospitals.hospital', 'name organization')
      .populate('regulatorDecision.regulator', 'name organization');
    
    if (!trial) {
      return res.status(404).json({
        success: false,
        error: 'Trial not found'
      });
    }

    // Get all reports for this trial
    const reports = await Report.find({ trial: trialId })
      .populate('patient', 'patientId age gender')
      .populate('hospital', 'name organization');

    // Verify trial initiation hash
    const trialData = {
      drugName: trial.drugName,
      description: trial.description,
      sampleSize: trial.sampleSize,
      manufacturer: trial.manufacturer._id.toString(),
      hospitals: trial.hospitals.map(h => h.hospital._id.toString()),
      timestamp: trial.createdAt.toISOString()
    };
    
    const calculatedTrialHash = generateHash(trialData);
    const trialHashValid = calculatedTrialHash === trial.initiationHash;

    // Verify hospital response hashes
    const hospitalResponses = trial.hospitals.map(hospital => {
      if (hospital.responseHash) {
        const hospitalData = {
          trialId: trial._id.toString(),
          hospitalId: hospital.hospital._id.toString(),
          response: hospital.status,
          timestamp: hospital.responseDate?.toISOString() || trial.createdAt.toISOString()
        };
        
        const calculatedHash = generateHash(hospitalData);
        return {
          hospital: hospital.hospital.name,
          response: hospital.status,
          hashValid: calculatedHash === hospital.responseHash
        };
      }
      return {
        hospital: hospital.hospital.name,
        response: hospital.status,
        hashValid: null // No response yet
      };
    });

    // Verify report hashes
    const reportVerifications = reports.map(report => {
      if (report.reportHash) {
        const reportData = {
          reportId: report._id.toString(),
          patientId: report.patient._id.toString(),
          trialId: report.trial.toString(),
          hospitalId: report.hospital._id.toString(),
          trialStartDate: report.trialStartDate,
          trialEndDate: report.trialEndDate,
          drugName: report.drugName,
          dosage: report.dosage,
          sideEffectSeverity: report.sideEffectSeverity,
          symptomImprovementScore: report.symptomImprovementScore,
          overallHealthStatus: report.overallHealthStatus,
          timestamp: report.createdAt.toISOString()
        };
        
        const calculatedHash = generateHash(reportData);
        return {
          reportId: report._id.toString(),
          patientId: report.patient.patientId,
          hospital: report.hospital.name,
          isFinal: report.isFinal,
          hashValid: calculatedHash === report.reportHash
        };
      }
      return {
        reportId: report._id.toString(),
        patientId: report.patient.patientId,
        hospital: report.hospital.name,
        isFinal: report.isFinal,
        hashValid: false
      };
    });

    // Verify regulator decision hash
    let regulatorDecisionValid = null;
    
    if (trial.regulatorDecision.decisionHash) {
      const decisionData = {
        trialId: trial._id.toString(),
        regulatorId: trial.regulatorDecision.regulator?._id.toString(),
        decision: trial.regulatorDecision.decision,
        comments: trial.regulatorDecision.comments,
        timestamp: trial.regulatorDecision.decisionDate?.toISOString()
      };
      
      const calculatedHash = generateHash(decisionData);
      regulatorDecisionValid = calculatedHash === trial.regulatorDecision.decisionHash;
    }

    // Compile audit results
    const auditResults = {
      trial: {
        id: trial._id,
        drugName: trial.drugName,
        manufacturer: trial.manufacturer.name,
        status: trial.status,
        initiationHashValid: trialHashValid
      },
      hospitalResponses,
      reports: reportVerifications,
      regulatorDecision: {
        decision: trial.regulatorDecision.decision,
        regulator: trial.regulatorDecision.regulator?.name,
        hashValid: regulatorDecisionValid
      },
      overallIntegrity: 
        trialHashValid && 
        hospitalResponses.every(h => h.hashValid !== false) &&
        reportVerifications.every(r => r.hashValid !== false) &&
        (regulatorDecisionValid !== false)
    };

    logger.info(`Audit completed for trial ${trialId}`);

    res.status(200).json({
      success: true,
      data: auditResults
    });
  } catch (error) {
    logger.error(`Trial audit error: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};