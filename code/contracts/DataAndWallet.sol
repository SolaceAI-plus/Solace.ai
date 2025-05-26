// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address recipient, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address sender, address recipient, uint256 amount) external returns (bool);
}

contract DataAndWalletInteraction {
    // 存储数据
    uint256 private storedData;
    string private storedString;
    
    // 记录合约创建者（管理员）
    address public owner;
    
    // 创建事件以记录交易
    event DataStored(uint256 newData);
    event StringStored(string newString);
    event TransferExecuted(address indexed from, address indexed to, uint256 value);
    
    // 设置合约的管理员
    constructor() {
        owner = msg.sender;  // 合约创建者为管理员
    }
    
    // 存储一个整数
    function storeData(uint256 _data) public {
        storedData = _data;
        emit DataStored(_data);
    }
    
    // 获取存储的整数
    function getStoredData() public view returns (uint256) {
        return storedData;
    }
    
    // 存储一个字符串
    function storeString(string memory _string) public {
        storedString = _string;
        emit StringStored(_string);
    }
    
    // 获取存储的字符串
    function getStoredString() public view returns (string memory) {
        return storedString;
    }
    
    // 转账 BNB
    function transferBNB(address payable _to, uint256 _amount) public payable {
        require(msg.sender == owner, "Only owner can send BNB");
        require(address(this).balance >= _amount, "Insufficient balance");
        
        _to.transfer(_amount);
        emit TransferExecuted(msg.sender, _to, _amount);
    }
    
    // 转账 BEP-20 Token
    function transferToken(address _token, address _to, uint256 _amount) public {
        require(msg.sender == owner, "Only owner can transfer tokens");
        IERC20 token = IERC20(_token);
        
        // 进行转账
        token.transfer(_to, _amount);
        emit TransferExecuted(msg.sender, _to, _amount);
    }
    
    // 获取合约余额
    function contractBalance() public view returns (uint256) {
        return address(this).balance;
    }
    
    // 合约接收 BNB
    receive() external payable {}
    
    // 获取余额
    function balanceOf(address _address) public view returns (uint256) {
        return address(_address).balance;
    }
}
