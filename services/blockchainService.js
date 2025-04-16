import { blockchainConfig } from '../config/blockchain.js';
import { generateHash, generateUniqueId } from '../utils/hashUtils.js';
import { logger } from '../utils/logger.js';

/**
 * Store a hash on the blockchain
 * @param {String} id - Unique identifier for the data
 * @param {Object} data - Data to hash and store
 * @returns {Object} - Transaction details
 */
export const storeHashOnBlockchain = async (id, data) => {
  try {
    const contract = blockchainConfig.getContract();
    const dataHash = generateHash(data);
    const uniqueId = id || generateUniqueId();
    
    // Store hash on blockchain
    const tx = await contract.storeHash(uniqueId, dataHash);
    const receipt = await tx.wait();
    
    logger.info(`Hash stored on blockchain. Transaction: ${receipt.transactionHash}`);
    
    return {
      success: true,
      transactionHash: receipt.transactionHash,
      blockNumber: receipt.blockNumber,
      dataHash,
      id: uniqueId
    };
  } catch (error) {
    logger.error(`Error storing hash on blockchain: ${error.message}`);
    throw new Error(`Blockchain error: ${error.message}`);
  }
};

/**
 * Verify data against a hash stored on the blockchain
 * @param {String} id - Unique identifier for the data
 * @param {Object} data - Data to verify
 * @returns {Boolean} - True if hash matches
 */
export const verifyHashOnBlockchain = async (id, data) => {
  try {
    const contract = blockchainConfig.getContract();
    const storedHash = await contract.getHash(id);
    const calculatedHash = generateHash(data);
    
    const isValid = storedHash === calculatedHash;
    
    logger.info(`Hash verification for ${id}: ${isValid ? 'Valid' : 'Invalid'}`);
    
    return {
      success: true,
      isValid,
      storedHash,
      calculatedHash
    };
  } catch (error) {
    logger.error(`Error verifying hash on blockchain: ${error.message}`);
    throw new Error(`Blockchain verification error: ${error.message}`);
  }
};

/**
 * Store trial initiation data on blockchain
 * @param {Object} trial - Trial data
 * @returns {Object} - Transaction details and hash
 */
export const storeTrialInitiation = async (trial) => {
  const trialId = trial._id.toString();
  const dataToHash = {
    drugName: trial.drugName,
    description: trial.description,
    sampleSize: trial.sampleSize,
    manufacturer: trial.manufacturer.toString(),
    hospitals: trial.hospitals.map(h => h.hospital.toString()),
    timestamp: new Date().toISOString()
  };
  
  return await storeHashOnBlockchain(`trial_init_${trialId}`, dataToHash);
};

/**
 * Store hospital response to trial on blockchain
 * @param {Object} trial - Trial data
 * @param {Object} hospital - Hospital data
 * @param {String} response - Response (accepted/rejected)
 * @returns {Object} - Transaction details and hash
 */
export const storeHospitalResponse = async (trial, hospital, response) => {
  const trialId = trial._id.toString();
  const hospitalId = hospital._id.toString();
  
  const dataToHash = {
    trialId,
    hospitalId,
    response,
    timestamp: new Date().toISOString()
  };
  
  return await storeHashOnBlockchain(`trial_resp_${trialId}_${hospitalId}`, dataToHash);
};

/**
 * Store patient report on blockchain
 * @param {Object} report - Report data
 * @returns {Object} - Transaction details and hash
 */
export const storePatientReport = async (report) => {
  const reportId = report._id.toString();
  const patientId = report.patient.toString();
  const trialId = report.trial.toString();
  
  const dataToHash = {
    reportId,
    patientId,
    trialId,
    hospitalId: report.hospital.toString(),
    trialStartDate: report.trialStartDate,
    trialEndDate: report.trialEndDate,
    drugName: report.drugName,
    dosage: report.dosage,
    sideEffectSeverity: report.sideEffectSeverity,
    symptomImprovementScore: report.symptomImprovementScore,
    overallHealthStatus: report.overallHealthStatus,
    timestamp: new Date().toISOString()
  };
  
  return await storeHashOnBlockchain(`report_${reportId}`, dataToHash);
};

/**
 * Store final report with AI analysis on blockchain
 * @param {Object} finalReport - Final report data
 * @param {Object} aiAnalysis - AI analysis data
 * @returns {Object} - Transaction details and hash
 */
export const storeFinalReport = async (finalReport, aiAnalysis) => {
  const reportId = finalReport._id.toString();
  const trialId = finalReport.trial.toString();
  
  const dataToHash = {
    reportId,
    trialId,
    hospitalId: finalReport.hospital.toString(),
    aiAnalysis,
    timestamp: new Date().toISOString()
  };
  
  return await storeHashOnBlockchain(`final_report_${reportId}`, dataToHash);
};

/**
 * Store regulator decision on blockchain
 * @param {Object} trial - Trial data
 * @param {Object} regulator - Regulator data
 * @param {String} decision - Decision (approved/rejected)
 * @param {String} comments - Decision comments
 * @returns {Object} - Transaction details and hash
 */
export const storeRegulatorDecision = async (trial, regulator, decision, comments) => {
  const trialId = trial._id.toString();
  const regulatorId = regulator._id.toString();
  
  const dataToHash = {
    trialId,
    regulatorId,
    decision,
    comments,
    timestamp: new Date().toISOString()
  };
  
  return await storeHashOnBlockchain(`regulator_decision_${trialId}`, dataToHash);
};