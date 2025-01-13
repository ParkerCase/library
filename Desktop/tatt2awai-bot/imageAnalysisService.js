// Create new file: imageAnalysisService.js

const fs = require('fs');
const path = require('path');
const logger = require('./logger');
const imageProcessor = require('./imageProcessor');
const dropboxManager = require('./dropboxManager');

class ImageAnalysisService {
  constructor() {
    this.analysisCache = new Map();
    this.processingQueue = new Set();
  }

  async analyzeImage(imageData, options = {}) {
    const tempPath = path.join('uploads', `temp_${Date.now()}`);
    try {
      fs.writeFileSync(tempPath, imageData);
      const analysis = await imageProcessor.processImage({
        path: tempPath,
        metadata: {
          ...options,
          timestamp: new Date().toISOString()
        }
      });

      return {
        analysis,
        metadata: {
          processed: true,
          timestamp: new Date().toISOString(),
          hasFeatures: !!analysis.tattooFeatures
        }
      };
    } finally {
      if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
    }
  }

  async findSimilarImages(analysisResult, options = {}) {
    const { threshold = 0.7, limit = 10 } = options;
    const dropboxFiles = await dropboxManager.fetchDropboxEntries('');
    const results = [];

    for (const file of dropboxFiles.result.entries) {
      if (this.isImageFile(file.path_lower)) {
        try {
          const fileData = await dropboxManager.downloadFile(file.path_lower);
          const similarity = await this.calculateSimilarity(
            analysisResult,
            await this.analyzeImage(fileData.result.fileBinary)
          );

          if (similarity >= threshold) {
            results.push({
              path: file.path_lower,
              similarity,
              metadata: {
                name: file.name,
                modified: file.server_modified,
                size: file.size
              }
            });
          }
        } catch (error) {
          logger.error('Error processing file for similarity:', {
            path: file.path_lower,
            error: error.message
          });
        }
      }
    }

    return results
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);
  }

  isImageFile(path) {
    const ext = path.toLowerCase().split('.').pop();
    return ['jpg', 'jpeg', 'png', 'gif', 'webp'].includes(ext);
  }

  async calculateSimilarity(analysis1, analysis2) {
    // Implement similarity calculation based on your needs
    // This is a simple example
    if (!analysis1?.analysis?.tattooFeatures || !analysis2?.analysis?.tattooFeatures) {
      return 0;
    }

    // Compare features, colors, etc.
    return 0.8; // Placeholder value
  }
}

module.exports = new ImageAnalysisService();
