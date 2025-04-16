import Trial from '../models/Trial.js';
import User from '../models/User.js';
import { logger } from '../utils/logger.js';
import { 
  storeTrialInitiation, 
  storeHospitalResponse 
} from '../services/blockchainService.js';

/**
 * @desc    Initiate a new drug trial
 * @route   POST /api/trials/initiate
 * @access  Private (Manufacturer only)
 */
export const initiateTrial = async (req, res) => {
  try {
    const { drugName, description, sampleSize, hospitalIds } = req.body;

    // Validate required fields
    if (!drugName || !description || !sampleSize || !hospitalIds || !hospitalIds.length) {
      return res.status(400).json({
        success: false,
        error: 'Please provide all required fields'
      });
    }

    // Verify all hospital IDs are valid and have hospital role
    const hospitals = await User.find({
      _id: { $in: hospitalIds },
      role: 'hospital'
    });

    if (hospitals.length !== hospitalIds.length) {
      return res.status(400).json({
        success: false,
        error: 'One or more hospital IDs are invalid'
      });
    }

    // Create trial object
    const trialData = {
      drugName,
      description,
      sampleSize,
      manufacturer: req.user.id,
      hospitals: hospitalIds.map(id => ({
        hospital: id,
        status: 'pending'
      })),
      status: 'initiated'
    };

    // Create trial in database (without hash initially)
    const trial = await Trial.create(trialData);

    // Store hash on blockchain
    const blockchainResult = await storeTrialInitiation(trial);

    // Update trial with hash
    trial.initiationHash = blockchainResult.dataHash;
    await trial.save();

    logger.info(`New trial initiated: ${trial._id} for drug ${drugName}`);

    res.status(201).json({
      success: true,
      data: trial,
      blockchain: {
        transactionHash: blockchainResult.transactionHash,
        blockNumber: blockchainResult.blockNumber
      }
    });
  } catch (error) {
    logger.error(`Trial initiation error: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

/**
 * @desc    Get all trials
 * @route   GET /api/trials
 * @access  Private (All roles)
 */
export const getTrials = async (req, res) => {
  try {
    let query = {};

    // Filter trials based on user role
    if (req.user.role === 'manufacturer') {
      query.manufacturer = req.user.id;
    } else if (req.user.role === 'hospital') {
      query['hospitals.hospital'] = req.user.id;
    }

    const trials = await Trial.find(query)
      .populate('manufacturer', 'name organization')
      .populate('hospitals.hospital', 'name organization');

    res.status(200).json({
      success: true,
      count: trials.length,
      data: trials
    });
  } catch (error) {
    logger.error(`Get trials error: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

/**
 * @desc    Get single trial
 * @route   GET /api/trials/:id
 * @access  Private (All roles with appropriate access)
 */
export const getTrial = async (req, res) => {
  try {
    const trial = await Trial.findById(req.params.id)
      .populate('manufacturer', 'name organization')
      .populate('hospitals.hospital', 'name organization')
      .populate('regulatorDecision.regulator', 'name organization');

    if (!trial) {
      return res.status(404).json({
        success: false,
        error: 'Trial not found'
      });
    }

    // Check if user has access to this trial
    const hasAccess = 
      req.user.role === 'regulator' ||
      trial.manufacturer.equals(req.user.id) ||
      trial.hospitals.some(h => h.hospital._id.equals(req.user.id));

    if (!hasAccess) {
      return res.status(403).json({
        success: false,
        error: 'Not authorized to access this trial'
      });
    }

    res.status(200).json({
      success: true,
      data: trial
    });
  } catch (error) {
    logger.error(`Get trial error: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};

/**
 * @desc    Hospital response to trial invitation
 * @route   POST /api/trials/:id/response
 * @access  Private (Hospital only)
 */
export const respondToTrial = async (req, res) => {
  try {
    const { response } = req.body;

    if (!response || !['accepted', 'rejected'].includes(response)) {
      return res.status(400).json({
        success: false,
        error: 'Please provide a valid response (accepted/rejected)'
      });
    }

    const trial = await Trial.findById(req.params.id);

    if (!trial) {
      return res.status(404).json({
        success: false,
        error: 'Trial not found'
      });
    }

    // Check if hospital is part of this trial
    const hospitalIndex = trial.hospitals.findIndex(
      h => h.hospital.toString() === req.user.id
    );

    if (hospitalIndex === -1) {
      return res.status(403).json({
        success: false,
        error: 'Hospital not invited to this trial'
      });
    }

    // Check if hospital has already responded
    if (trial.hospitals[hospitalIndex].status !== 'pending') {
      return res.status(400).json({
        success: false,
        error: 'Hospital has already responded to this trial'
      });
    }

    // Store response on blockchain
    const blockchainResult = await storeHospitalResponse(
      trial,
      { _id: req.user.id },
      response
    );

    // Update trial with hospital response
    trial.hospitals[hospitalIndex].status = response;
    trial.hospitals[hospitalIndex].responseDate = new Date();
    trial.hospitals[hospitalIndex].responseHash = blockchainResult.dataHash;

    // If at least one hospital accepted, update trial status
    if (response === 'accepted' && trial.status === 'initiated') {
      trial.status = 'in_progress';
    }

    await trial.save();

    logger.info(`Hospital ${req.user.id} ${response} trial ${trial._id}`);

    res.status(200).json({
      success: true,
      data: trial,
      blockchain: {
        transactionHash: blockchainResult.transactionHash,
        blockNumber: blockchainResult.blockNumber
      }
    });
  } catch (error) {
    logger.error(`Trial response error: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
};