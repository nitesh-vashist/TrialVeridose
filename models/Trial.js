import mongoose from 'mongoose';

const TrialSchema = new mongoose.Schema({
  drugName: {
    type: String,
    required: [true, 'Please add a drug name'],
    trim: true
  },
  description: {
    type: String,
    required: [true, 'Please add a description']
  },
  sampleSize: {
    type: Number,
    required: [true, 'Please specify the sample size'],
    min: [1, 'Sample size must be at least 1']
  },
  manufacturer: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  trialCode: {
    type: String,
    unique: true,
    sparse: true // Allows multiple nulls, avoids the duplicate key issue
  },
  hospitals: [{
    hospital: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    status: {
      type: String,
      enum: ['pending', 'accepted', 'rejected'],
      default: 'pending'
    },
    responseDate: {
      type: Date
    },
    responseHash: {
      type: String
    }
  }],
  status: {
    type: String,
    enum: ['initiated', 'in_progress', 'completed', 'approved', 'rejected'],
    default: 'initiated'
  },
  initiationHash: {
    type: String,
    required: false,
    default: null
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  completedAt: {
    type: Date
  },
  regulatorDecision: {
    decision: {
      type: String,
      enum: ['pending', 'approved', 'rejected'],
      default: 'pending'
    },
    regulator: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'User'
    },
    decisionDate: {
      type: Date
    },
    decisionHash: {
      type: String
    },
    comments: {
      type: String
    }
  }
});

export default mongoose.model('Trial', TrialSchema);