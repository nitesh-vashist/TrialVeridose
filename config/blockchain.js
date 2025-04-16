import { ethers } from 'ethers';
import { logger } from '../utils/logger.js';
import dotenv from 'dotenv';
dotenv.config();
console.log('RPC:', process.env.ETHEREUM_RPC_URL);
console.log('Contract:', process.env.CONTRACT_ADDRESS);


// Initialize provider based on environment
const getProvider = () => {
  if (process.env.NODE_ENV === 'production') {
    // return new ethers.providers.JsonRpcProvider(process.env.ETHEREUM_RPC_URL);
    return new ethers.JsonRpcProvider(process.env.ETHEREUM_RPC_URL);
  } else {
    // return new ethers.providers.JsonRpcProvider(process.env.ETHEREUM_RPC_URL || 'http://localhost:8545');
    return new ethers.JsonRpcProvider(process.env.ETHEREUM_RPC_URL || 'http://localhost:8545');
  }
};

// Initialize wallet with private key
const getWallet = () => {
  const provider = getProvider();
  return new ethers.Wallet(process.env.ETHEREUM_PRIVATE_KEY, provider);
};

// Contract ABI and address
const contractABI = [
    
        {
            "anonymous": false,
            "inputs": [
                {
                    "indexed": true,
                    "internalType": "string",
                    "name": "id",
                    "type": "string"
                },
                {
                    "indexed": false,
                    "internalType": "string",
                    "name": "hash",
                    "type": "string"
                }
            ],
            "name": "HashStored",
            "type": "event"
        },
        {
            "inputs": [
                {
                    "internalType": "string",
                    "name": "id",
                    "type": "string"
                },
                {
                    "internalType": "string",
                    "name": "dataHash",
                    "type": "string"
                }
            ],
            "name": "storeHash",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function"
        },
        {
            "inputs": [
                {
                    "internalType": "string",
                    "name": "id",
                    "type": "string"
                }
            ],
            "name": "getHash",
            "outputs": [
                {
                    "internalType": "string",
                    "name": "",
                    "type": "string"
                }
            ],
            "stateMutability": "view",
            "type": "function"
        }
    
];

const getContract = () => {
  const wallet = getWallet();
  return new ethers.Contract(
    process.env.CONTRACT_ADDRESS,
    contractABI,
    wallet
  );
};

export const blockchainConfig = {
  getProvider,
  getWallet,
  getContract
};