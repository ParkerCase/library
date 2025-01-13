const express = require('express');
const multer = require('multer');
const fs = require('fs');

// Basic express setup
const app = express();

// Basic multer setup
const upload = multer({ dest: 'uploads/' });

// Single debugging endpoint
app.post('/debug-upload', upload.single('image'), (req, res) => {
    console.log('Request received');
    
    if (!req.file) {
        console.log('No file received');
        return res.status(400).send('No file uploaded');
    }

    console.log('File received:', {
        filename: req.file.filename,
        originalname: req.file.originalname,
        size: req.file.size
    });

    return res.status(200).json({
        message: 'Upload successful',
        file: {
            filename: req.file.filename,
            originalname: req.file.originalname,
            size: req.file.size
        }
    });
});

// Start server
const PORT = 4003;
app.listen(PORT, () => {
    console.log(`Debug server running on port ${PORT}`);
});
