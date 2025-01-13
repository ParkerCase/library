const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Create necessary directories (only creating cache and logs since uploads exists)
const dirs = [
  'cache',
  'logs'
];

dirs.forEach(dir => {
  const dirPath = path.join(__dirname, dir);
  if (!fs.existsSync(dirPath)) {
    fs.mkdirSync(dirPath, { recursive: true });
    console.log(`Created directory: ${dirPath}`);
  } else {
    console.log(`Directory already exists: ${dirPath}`);
  }
});

// Install required dependencies
const dependencies = [
  '@supabase/supabase-js',
  'express-rate-limit',
  'dropbox',
  '@google-cloud/vision',
  'openai',
  'multer',
  'axios'
];

console.log('\nInstalling/updating dependencies...');
dependencies.forEach(dep => {
  try {
    console.log(`Installing ${dep}...`);
    execSync(`npm install ${dep}`, { stdio: 'inherit' });
    console.log(`✓ Successfully installed ${dep}`);
  } catch (error) {
    console.error(`× Failed to install ${dep}:`, error.message);
  }
});

// Verify .env file exists
const envPath = path.join(__dirname, '.env');
if (fs.existsSync(envPath)) {
  console.log('\n✓ .env file found');
} else {
  console.error('× .env file not found. Please create it with the required credentials.');
  process.exit(1);
}

console.log('\nSetup complete! Next steps:');
console.log('1. Verify all dependencies were installed correctly');
console.log('2. Make sure all environment variables are set in .env');
console.log('3. Start the server with: node server.js');
