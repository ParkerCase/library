const fs = require('fs');
const path = require('path');
const logger = require('./logger');
const imageProcessor = require('./imageProcessor');

const imageSimilarity = {
  processedImages: new Map(),
  featureCache: new Map(),
  
  async processImage(path) {
    // Check cache first
    if (this.processedImages.has(path)) {
      return this.processedImages.get(path);
    }

    try {
      const dropboxManager = require('./dropboxManager');
      const file = await dropboxManager.downloadFile(path);
      
      if (!file?.result?.fileBinary) {
        throw new Error('Failed to download image');
      }

      const tempPath = `temp_${Date.now()}_${path.split('/').pop()}`;
      fs.writeFileSync(tempPath, file.result.fileBinary);
      
      try {
        const analysis = await imageProcessor.processImage({ path: tempPath });
        const features = {
          analysis,
          path,
          timestamp: Date.now()
        };
        
        this.processedImages.set(path, features);
        return features;
      } finally {
        if (fs.existsSync(tempPath)) {
          fs.unlinkSync(tempPath);
        }
      }
    } catch (error) {
      logger.error('Error processing image:', { path, error: error.message });
      throw error;
    }
  },

  async findSimilar(targetFeatures, allImages, options = {}) {
    const {
      limit = 10,
      minSimilarity = 0.6,
      includeMetadata = true
    } = options;

    try {
      const results = [];
      
      for (const image of allImages) {
        try {
          const features = await this.processImage(image.path_lower);
          const similarity = await this.calculateSimilarity(targetFeatures, features);
          
          if (similarity >= minSimilarity) {
            results.push({
              path: image.path_lower,
              similarity,
              features: includeMetadata ? features : undefined,
              metadata: {
                ...image,
                processed: new Date().toISOString()
              }
            });
          }
        } catch (imageError) {
          logger.warn('Error processing comparison image:', {
            path: image.path_lower,
            error: imageError.message
          });
          continue;
        }
      }

      return results
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, limit);
    } catch (error) {
      logger.error('Error in findSimilar:', error);
      throw error;
    }
  },

  async calculateSimilarity(features1, features2) {
    try {
      let score = 0;
      let weights = 0;

      // Visual feature similarity
      if (features1.analysis && features2.analysis) {
        const visualScore = this.compareVisualFeatures(
          features1.analysis,
          features2.analysis
        );
        score += visualScore * 3;
        weights += 3;
      }

      // Color similarity
      if (features1.analysis?.colors && features2.analysis?.colors) {
        const colorScore = this.compareColors(
          features1.analysis.colors,
          features2.analysis.colors
        );
        score += colorScore * 2;
        weights += 2;
      }

      // Content similarity
      if (features1.analysis?.labels && features2.analysis?.labels) {
        const contentScore = this.compareContent(
          features1.analysis.labels,
          features2.analysis.labels
        );
        score += contentScore;
        weights += 1;
      }

      return weights > 0 ? score / weights : 0;
    } catch (error) {
      logger.error('Error calculating similarity:', error);
      return 0;
    }
  },

  compareVisualFeatures(analysis1, analysis2) {
    try {
      const tattooFeatures1 = analysis1.tattooFeatures;
      const tattooFeatures2 = analysis2.tattooFeatures;

      if (!tattooFeatures1 || !tattooFeatures2) return 0;

      let score = 0;
      let weights = 0;

      // Compare ink colors
      if (tattooFeatures1.inkColors && tattooFeatures2.inkColors) {
        const colorSimilarity = this.compareInkColors(
          tattooFeatures1.inkColors,
          tattooFeatures2.inkColors
        );
        score += colorSimilarity * 2;
        weights += 2;
      }

      // Compare density
      if (tattooFeatures1.detailedAnalysis && tattooFeatures2.detailedAnalysis) {
        const densityDiff = Math.abs(
          tattooFeatures1.detailedAnalysis.density -
          tattooFeatures2.detailedAnalysis.density
        );
        score += (1 - densityDiff) * 1.5;
        weights += 1.5;
      }

      // Compare placement
      if (tattooFeatures1.detailedAnalysis?.placement === 
          tattooFeatures2.detailedAnalysis?.placement) {
        score += 1;
        weights += 1;
      }

      return weights > 0 ? score / weights : 0;
    } catch (error) {
      logger.error('Error comparing visual features:', error);
      return 0;
    }
  },

  compareInkColors(colors1, colors2) {
    try {
      if (!colors1.length || !colors2.length) return 0;
      
      let totalSimilarity = 0;
      let matches = 0;

      for (const color1 of colors1) {
        for (const color2 of colors2) {
          const similarity = this.calculateColorSimilarity(
            color1.rgb,
            color2.rgb
          );
          if (similarity > 0.8) {
            totalSimilarity += similarity;
            matches++;
          }
        }
      }

      return matches > 0 ? totalSimilarity / matches : 0;
    } catch (error) {
      logger.error('Error comparing ink colors:', error);
      return 0;
    }
  },

  calculateColorSimilarity(rgb1, rgb2) {
    try {
      const [r1, g1, b1] = this.parseRgb(rgb1);
      const [r2, g2, b2] = this.parseRgb(rgb2);
      
      const distance = Math.sqrt(
        Math.pow(r1 - r2, 2) +
        Math.pow(g1 - g2, 2) +
        Math.pow(b1 - b2, 2)
      );
      
      return 1 - (distance / (Math.sqrt(3) * 255));
    } catch (error) {
      logger.error('Error calculating color similarity:', error);
      return 0;
    }
  },

  parseRgb(rgbString) {
    const matches = rgbString.match(/\d+/g);
    return matches ? matches.map(Number) : [0, 0, 0];
  },

  compareContent(labels1, labels2) {
    try {
      const set1 = new Set(labels1.map(l => l.description.toLowerCase()));
      const set2 = new Set(labels2.map(l => l.description.toLowerCase()));
      
      const intersection = new Set([...set1].filter(x => set2.has(x)));
      const union = new Set([...set1, ...set2]);
      
      return intersection.size / union.size;
    } catch (error) {
      logger.error('Error comparing content:', error);
      return 0;
    }
  },

  compareColors(colors1, colors2) {
    try {
      if (!colors1 || !colors2) return 0;
      
      let totalSimilarity = 0;
      let weights = 0;

      // Compare dominant colors
      for (const color1 of colors1) {
        for (const color2 of colors2) {
          const similarity = this.calculateColorSimilarity(color1.rgb, color2.rgb);
          const weight = color1.score * color2.score;
          totalSimilarity += similarity * weight;
          weights += weight;
        }
      }

      return weights > 0 ? totalSimilarity / weights : 0;
    } catch (error) {
      logger.error('Error comparing colors:', error);
      return 0;
    }
  }
};

module.exports = imageSimilarity;
