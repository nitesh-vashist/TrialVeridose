import mongoose from 'mongoose';

const ReportSchema = new mongoose.Schema({
  patient: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Patient',
    required: true
  },
  trial: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Trial',
    required: true
  },
  hospital: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  trialStartDate: {
    type: Date,
    required: [true, 'Please add a trial start date']
  },
  trialEndDate: {
    type: Date,
    required: [true, 'Please add a trial end date'],
    validate: {
      validator: function(value) {
        return value >= this.trialStartDate;
      },
      message: 'End date must be after or equal to start date'
    }
  },
  drugName: {
    type: String,
    required: [true, 'Please add a drug name']
  },
  dosage: {
    type: String,
    required: [true, 'Please add dosage information']
  },
  preExistingConditions: {
    type: [String],
    default: []
  },
  observedSideEffects: {
    type: [String],
    default: []
  },
  sideEffectSeverity: {
    type: String,
    enum: ['None', 'Mild', 'Moderate', 'Severe'],
    required: true
  },
  symptomImprovementScore: {
    type: Number,
    required: true,
    min: 0,
    max: 10
  },
  overallHealthStatus: {
    type: String,
    enum: ['Worse', 'Same', 'Improved'],
    required: true
  },
  doctorNotes: {
    type: String
  },
  reportHash: {
    type: String,
    // required: true
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  isFinal: {
    type: Boolean,
    default: false
  },
  aiAnalysis: {
    type: Object,
    default: null
  },
  aiAnalysisHash: {
    type: String
  }
});

// Create a compound index for trial and patient to ensure uniqueness
ReportSchema.index({ trial: 1, patient: 1 }, { unique: true });

export default mongoose.model('Report', ReportSchema);