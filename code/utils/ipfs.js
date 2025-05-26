const { create } = require('ipfs-http-client');
const projectId = process.env.IPFS_PROJECT_ID;
const projectSecret = process.env.IPFS_API_SECRET;
const auth = 'Basic ' + Buffer.from(projectId + ':' + projectSecret).toString('base64');

const client = create({
  host: 'ipfs.infura.io',
  port: 5001,
  protocol: 'https',
  headers: {
    authorization: auth,
  },
});

async function uploadToIpfs(content) {
  const { path } = await client.add(content);
  return path;
}

module.exports = { uploadToIpfs };