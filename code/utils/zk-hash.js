const crypto = require('crypto');
module.exports = function hash(data) {
  return ethers.utils.keccak256(Buffer.from(data));
};