require('dotenv').config();
const { ethers } = require('hardhat');
const { uploadToIpfs } = require('../utils/ipfs');
const zkHash = require('../utils/zk-hash');

async function main() {
  const [deployer] = await ethers.getSigners();
  const contract = await ethers.getContractAt("DataMarketplace", "");

  const data = JSON.stringify({ id: "123", value: 42 });
  const ipfsCid = await uploadToIpfs(data);
  const hash = zkHash(data);

  const tx = await contract.listData(ipfsCid, hash, ethers.utils.parseEther("0.05"));
  await tx.wait();
  console.log("data upload:", ipfsCid);
}

main();