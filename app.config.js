module.exports = {
  apps : [{
    name   : 'nas_chain_miner',
    script : 'neurons/miner.py',
    interpreter: 'python3',
    min_uptime: '5m',
    max_restarts: '5',
    args: ['--netuid','123','--wallet.name','miner1','--wallet.hotkey','default','--logging.debug','--axon.port','5001','--dht.port','5002','--dht.announce_ip','24.5.90.8','--dht.announce_ip','24.5.90.8','--genomaster.ip','http://51.161.12.128','--genomaster.port','5000','--subtensor.network','test']
  }]
}
