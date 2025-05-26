// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract DataMarketplace {
    struct DataItem {
        address seller;
        string ipfsCid;
        bytes32 zkHash;
        uint256 price;
        bool isSold;
    }

    mapping(uint256 => DataItem) public dataItems;
    mapping(uint256 => address) public buyers;
    uint256 public itemCount;

    event DataListed(uint256 itemId, address seller, uint256 price);
    event DataPurchased(uint256 itemId, address buyer);

    function listData(string memory ipfsCid, bytes32 zkHash, uint256 price) external {
        dataItems[itemCount] = DataItem({
            seller: msg.sender,
            ipfsCid: ipfsCid,
            zkHash: zkHash,
            price: price,
            isSold: false
        });
        emit DataListed(itemCount, msg.sender, price);
        itemCount++;
    }

    function buyData(uint256 itemId) external payable {
        DataItem storage item = dataItems[itemId];
        require(!item.isSold, "Already sold");
        require(msg.value == item.price, "Incorrect payment");

        item.isSold = true;
        buyers[itemId] = msg.sender;
        payable(item.seller).transfer(msg.value);

        emit DataPurchased(itemId, msg.sender);
    }

    function getDataCid(uint256 itemId) external view returns (string memory) {
        require(buyers[itemId] == msg.sender, "Not authorized");
        return dataItems[itemId].ipfsCid;
    }
}
