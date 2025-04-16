import mongoose from 'mongoose';

const PatientSchema = new mongoose.Schema({
  patientId: {
    type: String,
    required: [true, 'Please add a patient ID'],
    unique: true,
    trim: true
  },
  age: {
    type: Number,
    required: [true, 'Please add age'],
    min: 0,
    max: 120
  },
  gender: {
    type: String,
    required: [true, 'Please specify gender'],
    enum: ['Male', 'Female', 'Other']
  },
  hospital: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  trial: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Trial',
    required: true
  },
  createdAt: {
    type: Date,
    default: Date.now
  }
});

export default mongoose.model('Patient', PatientSchema);