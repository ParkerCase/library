module.exports = {
  apps: [{
    name: "tatt2awai-bot",
    script: "server.js",
    instances: 1,
    exec_mode: "fork",
    watch: false,
    autorestart: true,
max_memory_restart: "450M",  // Changed from 2G
    node_args: "--max-old-space-size=400",  // Changed from 8192
    env: {
      NODE_ENV: "production",
      UV_THREADPOOL_SIZE: 4  // Changed from 64 to match your system
    },
    log_date_format: "YYYY-MM-DD HH:mm:ss Z",
    error_file: "logs/error.log",
    out_file: "logs/output.log",
    merge_logs: true,
    time: true
  }]
};
