const hre = require("hardhat");

async function main() {
  const Contract = await hre.ethers.getContractFactory("DataMarketplace");
  const contract = await Contract.deploy();
  await contract.deployed();
  console.log(`Contract deployed at: ${contract.address}`);
}

main();