import crypto from 'crypto';

/**
 * Generate a SHA-256 hash of the provided data
 * @param {Object} data - Data to hash
 * @returns {String} - Hex string of the hash
 */
export const generateHash = (data) => {
  const stringData = typeof data === 'object' ? JSON.stringify(data) : String(data);
  return crypto.createHash('sha256').update(stringData).digest('hex');
};

/**
 * Verify if the provided data matches the stored hash
 * @param {Object} data - Data to verify
 * @param {String} storedHash - Previously stored hash
 * @returns {Boolean} - True if hash matches
 */
export const verifyHash = (data, storedHash) => {
  const generatedHash = generateHash(data);
  return generatedHash === storedHash;
};

/**
 * Generate a timestamp for blockchain transactions
 * @returns {String} - ISO timestamp
 */
export const generateTimestamp = () => {
  return new Date().toISOString();
};

/**
 * Generate a unique ID for blockchain transactions
 * @param {String} prefix - Optional prefix for the ID
 * @returns {String} - Unique ID
 */
export const generateUniqueId = (prefix = '') => {
  const timestamp = Date.now().toString(36);
  const randomStr = Math.random().toString(36).substring(2, 8);
  return `${prefix}${timestamp}${randomStr}`;
};