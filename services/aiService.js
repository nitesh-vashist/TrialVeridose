import fetch from 'node-fetch';
import { logger } from '../utils/logger.js';

/**
 * Analyze a final report using AI
 * @param {Object} finalReport - The final report data
 * @returns {Object} - AI analysis results
 */
export const analyzeReport = async (finalReport) => {
  try {
    const AI_SERVICE_URL = process.env.AI_SERVICE_URL;
    
    if (!AI_SERVICE_URL) {
      logger.error('AI_SERVICE_URL environment variable not set');
      throw new Error('AI service URL not configured');
    }
    
    // Prepare data for AI analysis
    // const analysisData = {
    //   trialId: finalReport.trial.toString(),
    //   drugName: finalReport.drugName,
    //   patientReports: finalReport.patientReports || [],
    //   aggregateData: {
    //     totalPatients: finalReport.totalPatients,
    //     averageImprovementScore: finalReport.averageImprovementScore,
    //     sideEffectDistribution: finalReport.sideEffectDistribution,
    //     healthStatusDistribution: finalReport.healthStatusDistribution
    //   }
    // };
    
    // Send data to AI service
    const response = await fetch(AI_SERVICE_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.AI_SERVICE_API_KEY}`
      },
      body: JSON.stringify(finalReport)
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      logger.error(`AI service error: ${response.status} - ${errorText}`);
      throw new Error(`AI service returned ${response.status}: ${errorText}`);
    }
    
    const analysisResult = await response.json();
    
    logger.info(`AI analysis completed for trial ${finalReport.trial}`);
    
    return {
      success: true,
      analysis: analysisResult
    };
  } catch (error) {
    logger.error(`Error in AI analysis: ${error.message}`);
    throw new Error(`AI analysis error: ${error.message}`);
  }
};

/**
 * Mock AI analysis for development/testing
 * @param {Object} finalReport - The final report data
 * @returns {Object} - Mock AI analysis results
 */
export const mockAnalyzeReport = (finalReport) => {
  logger.info(`Generating mock AI analysis for trial ${finalReport.trial}`);
  
  // Calculate effectiveness score based on improvement scores
  const effectivenessScore = finalReport.averageImprovementScore / 10 * 100;
  
  // Calculate safety score based on side effect severity
  let safetyScore = 100;
  if (finalReport.sideEffectDistribution) {
    if (finalReport.sideEffectDistribution.Severe > 0) {
      safetyScore -= 40;
    } else if (finalReport.sideEffectDistribution.Moderate > 0) {
      safetyScore -= 20;
    } else if (finalReport.sideEffectDistribution.Mild > 0) {
      safetyScore -= 10;
    }
  }
  
  // Generate recommendation based on scores
  let recommendation = '';
  if (effectivenessScore > 70 && safetyScore > 80) {
    recommendation = 'Strongly recommend approval. The drug shows high effectiveness with minimal safety concerns.';
  } else if (effectivenessScore > 50 && safetyScore > 60) {
    recommendation = 'Recommend approval with monitoring. The drug shows moderate effectiveness with acceptable safety profile.';
  } else if (effectivenessScore > 30 && safetyScore > 40) {
    recommendation = 'Recommend additional trials. The drug shows limited effectiveness or has safety concerns that need further investigation.';
  } else {
    recommendation = 'Do not recommend approval. The drug shows poor effectiveness or significant safety concerns.';
  }
  
  return {
    success: true,
    analysis: {
      drugName: finalReport.drugName,
      trialId: finalReport.trial.toString(),
      effectivenessScore: Math.round(effectivenessScore),
      safetyScore: Math.round(safetyScore),
      patientImpact: {
        improved: finalReport.healthStatusDistribution?.Improved || 0,
        unchanged: finalReport.healthStatusDistribution?.Same || 0,
        worsened: finalReport.healthStatusDistribution?.Worse || 0
      },
      sideEffectAnalysis: {
        severity: Object.entries(finalReport.sideEffectDistribution || {})
          .map(([severity, count]) => ({ severity, count })),
        commonSideEffects: finalReport.commonSideEffects || []
      },
      recommendation,
      timestamp: new Date().toISOString()
    }
  };
};