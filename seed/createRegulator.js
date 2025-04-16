import mongoose from 'mongoose';
import dotenv from 'dotenv';
import User from '../models/User.js';
import { connectDB } from '../config/db.js';

// Load environment variables
dotenv.config();

// Connect to database
connectDB();

const createRegulator = async () => {
  try {
    // Check if regulator already exists
    const existingRegulator = await User.findOne({ 
      email: process.env.REGULATOR_EMAIL || 'regulator@veridose.gov' 
    });

    if (existingRegulator) {
      console.log('Regulator account already exists');
      process.exit(0);
    }

    // Create regulator account
    const regulator = await User.create({
      name: process.env.REGULATOR_NAME || 'Regulatory Authority',
      email: process.env.REGULATOR_EMAIL || 'regulator@veridose.gov',
      password: process.env.REGULATOR_PASSWORD || 'password123',
      role: 'regulator',
      organization: process.env.REGULATOR_ORG || 'Drug Regulatory Authority'
    });

    console.log(`Regulator account created: ${regulator.email}`);
    process.exit(0);
  } catch (error) {
    console.error(`Error creating regulator: ${error.message}`);
    process.exit(1);
  }
};

createRegulator();