// test-server.js
const express = require('express');
const multer = require('multer');
const app = express();

// Basic upload setup
const upload = multer({ dest: 'uploads/' });

// Test route
app.get('/test', (req, res) => {
  console.log('Test endpoint hit');
  res.json({ ok: true });
});

// Simple upload route
app.post('/upload', upload.single('image'), (req, res) => {
  console.log('Upload received');
  res.json({ received: true });
});

// Start server
app.listen(4000, () => {
  console.log('Test server running on port 4000');
});
