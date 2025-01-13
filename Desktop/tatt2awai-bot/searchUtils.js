const logger = require('./logger');
const imageProcessor = require('./imageProcessor');
const dropboxManager = require('./dropboxManager');

const searchUtils = {
    calculateImageSimilarity: async function(image1, image2) {
        const factors = {
            visualSimilarity: 0.4,
            tattooFeatures: 0.3,
            locationMatch: 0.2,
            colorProfile: 0.1
        };

        const scores = {
            visualSimilarity: await this.compareVisualFeatures(image1, image2),
            tattooFeatures: this.compareTattooFeatures(image1, image2),
            locationMatch: this.compareLocations(image1, image2),
            colorProfile: this.compareColorProfiles(image1, image2)
        };

        return Object.entries(factors).reduce((total, [factor, weight]) => 
            total + (scores[factor] * weight), 0);
    },

    compareVisualFeatures: async function(image1, image2) {
        const featureScore = await this.compareFeatures(image1, image2);
        const colorScore = await this.compareColors(image1, image2);
        const patternScore = await this.comparePatterns(image1, image2);
        
        return (featureScore * 0.4 + colorScore * 0.3 + patternScore * 0.3);
    },

    compareFeatures: async function(image1, image2) {
        const features1 = image1.analysis?.tattooFeatures || {};
        const features2 = image2.analysis?.tattooFeatures || {};
        
        let score = 0;
        let totalWeight = 0;
        
        if (features1.isTattoo === features2.isTattoo) score += 1;
        if (features1.skinTone === features2.skinTone) score += 0.5;
        if (features1.tattooStyle === features2.tattooStyle) score += 1;
        
        totalWeight = 2.5;
        
        return score / totalWeight;
    },

    compareColors: async function(image1, image2) {
        const colors1 = image1.analysis?.colors || [];
        const colors2 = image2.analysis?.colors || [];
        
        if (!colors1.length || !colors2.length) return 0;
        
        const histogram1 = this.buildColorHistogram(colors1);
        const histogram2 = this.buildColorHistogram(colors2);
        
        return this.compareHistograms(histogram1, histogram2);
    },

    buildColorHistogram: function(colors) {
        return colors.reduce((hist, color) => {
            const key = this.quantizeColor(color.rgb);
            hist[key] = (hist[key] || 0) + color.score;
            return hist;
        }, {});
    },

    quantizeColor: function(rgb) {
        const r = Math.floor(rgb.red / 32) * 32;
        const g = Math.floor(rgb.green / 32) * 32;
        const b = Math.floor(rgb.blue / 32) * 32;
        return `${r},${g},${b}`;
    },

    compareHistograms: function(hist1, hist2) {
        const allKeys = new Set([...Object.keys(hist1), ...Object.keys(hist2)]);
        let similarity = 0;
        
        allKeys.forEach(key => {
            const val1 = hist1[key] || 0;
            const val2 = hist2[key] || 0;
            similarity += Math.min(val1, val2);
        });
        
        return similarity;
    },

    comparePatterns: async function(image1, image2) {
        if (!image1.analysis?.tattooFeatures?.detailedAnalysis || 
            !image2.analysis?.tattooFeatures?.detailedAnalysis) {
            return 0;
        }
        
        const analysis1 = image1.analysis.tattooFeatures.detailedAnalysis;
        const analysis2 = image2.analysis.tattooFeatures.detailedAnalysis;
        
        const densityScore = 1 - Math.abs(analysis1.density - analysis2.density);
        const complexityScore = 1 - Math.abs(analysis1.complexity - analysis2.complexity);
        const edgeScore = 1 - Math.abs(analysis1.edgeSharpness - analysis2.edgeSharpness);
        
        return (densityScore + complexityScore + edgeScore) / 3;
    },

    findImageSequences: async function(images, criteria = {}) {
        const sequences = [];
        const processed = new Set();

        for (const image of images) {
            if (processed.has(image.path)) continue;

            const sequence = await this.buildImageSequence(image, images, criteria);
            if (sequence.length > 1) {
                sequences.push({
                    images: sequence,
                    metrics: await this.calculateSequenceMetrics(sequence),
                    confidence: await this.calculateSequenceConfidence(sequence)
                });
                sequence.forEach(img => processed.add(img.path));
            }
        }

        return sequences;
    },

    buildImageSequence: async function(startImage, allImages, criteria = {}) {
        try {
            const sequence = [startImage];
            const remainingImages = allImages.filter(img => img.path !== startImage.path);
            
            const scoredImages = await Promise.all(remainingImages.map(async img => ({
                image: img,
                similarity: await this.calculateImageSimilarity(startImage, img),
                timestamp: new Date(img.metadata?.timestamp || 0)
            })));
            
            scoredImages.sort((a, b) => {
                const similarityDiff = b.similarity - a.similarity;
                if (Math.abs(similarityDiff) > 0.1) return similarityDiff;
                return a.timestamp - b.timestamp;
            });
            
            for (const scored of scoredImages) {
                if (scored.similarity > 0.7) {
                    sequence.push(scored.image);
                }
            }
            
            return sequence;
            
        } catch (error) {
            logger.error('Error building image sequence:', error);
            return [startImage];
        }
    },

    calculateSequenceMetrics: async function(sequence) {
        return {
            totalProgress: await this.calculateTotalProgress(sequence),
            progressionRate: this.calculateProgressionRate(sequence),
            consistencyScore: this.calculateConsistencyScore(sequence),
            qualityMetrics: await this.calculateQualityMetrics(sequence)
        };
    },

    enhancedSearchWithContext: async function(query, context = {}) {
        const results = {
            exact: [],
            similar: [],
            related: [],
            sequences: []
        };

        try {
            if (context.currentImage) {
                const exactMatches = await this.findExactMatches(context.currentImage);
                results.exact = exactMatches;
            }

            const similarImages = await this.findSimilarImages(query, context);
            results.similar = similarImages;

            const relatedImages = await this.findRelatedImages(query, context);
            results.related = relatedImages;

            if (results.exact.length || results.similar.length) {
                const baseImage = results.exact[0] || results.similar[0];
                const sequences = await this.buildProgressionSequences(baseImage, 
                    [...results.similar, ...results.related]);
                results.sequences = sequences;
            }

            return {
                results,
                metadata: {
                    query,
                    timestamp: new Date().toISOString(),
                    context: {
                        hasCurrentImage: !!context.currentImage,
                        totalFound: Object.values(results).flat().length,
                        sequencesFound: results.sequences.length
                    }
                }
            };
        } catch (error) {
            logger.error('Enhanced search failed:', error);
            throw error;
        }
    },

    findExactMatches: async function(image) {
        const fingerprint = await imageProcessor.generateImageFingerprint(image);
        const allImages = await dropboxManager.fetchDropboxEntries('');
        
        return allImages.result.entries.filter(async entry => {
            const entryFingerprint = await imageProcessor.generateImageFingerprint(entry);
            return fingerprint === entryFingerprint;
        });
    },

    findSimilarImages: async function(query, context) {
        const allImages = await dropboxManager.fetchDropboxEntries('');
        const imageEntries = allImages.result.entries.filter(entry => 
            ['.jpg', '.jpeg', '.png', '.gif', '.webp'].some(ext => 
                entry.path_lower.endsWith(ext)
            )
        );

        const similarityPromises = imageEntries.map(async entry => {
            const similarity = context.currentImage ? 
                await this.calculateImageSimilarity(context.currentImage, entry) :
                await this.calculateTextSimilarity(query, entry);
                
            return {
                entry,
                similarity,
                analysis: await imageProcessor.processDropboxImage(entry.path_lower)
            };
        });

        const results = await Promise.all(similarityPromises);
        return results
            .filter(result => result.similarity > 0.7)
            .sort((a, b) => b.similarity - a.similarity);
    },

    findRelatedImages: async function(query, context) {
        const allImages = await dropboxManager.fetchDropboxEntries('');
        
        const targetFeatures = context.currentImage ? 
            await imageProcessor.enhancedAnalyzeTattooFeatures(context.currentImage) :
            await this.extractFeaturesFromQuery(query);

        const relatedPromises = allImages.result.entries.map(async entry => {
            const analysis = await imageProcessor.processDropboxImage(entry.path_lower);
            const relationScore = this.calculateRelationScore(targetFeatures, analysis);
            
            return {
                entry,
                relationScore,
                analysis
            };
        });

        const results = await Promise.all(relatedPromises);
        return results
            .filter(result => result.relationScore > 0.5)
            .sort((a, b) => b.relationScore - a.relationScore);
    },

    calculateRelationScore: function(targetFeatures, comparisonAnalysis) {
        let score = 0;
        let weights = 0;

        if (targetFeatures.tattooStyle === comparisonAnalysis.tattooFeatures.tattooStyle) {
            score += 0.3;
            weights += 0.3;
        }

        if (targetFeatures.placement === comparisonAnalysis.tattooFeatures.placement) {
            score += 0.3;
            weights += 0.3;
        }

        const colorSimilarity = this.compareColorPalettes(
            targetFeatures.inkColors,
            comparisonAnalysis.tattooFeatures.inkColors
        );
        score += colorSimilarity * 0.4;
        weights += 0.4;

        return weights > 0 ? score / weights : 0;
    },

    compareColorPalettes: function(colors1, colors2) {
        if (!colors1 || !colors2 || !colors1.length || !colors2.length) {
            return 0;
        }

        const palette1 = this.normalizePalette(colors1);
        const palette2 = this.normalizePalette(colors2);

        let similarity = 0;
        let count = 0;

        for (const color1 of palette1) {
            const bestMatch = palette2.reduce((best, color2) => {
                const match = this.calculateColorMatch(color1, color2);
                return match > best ? match : best;
            }, 0);
            similarity += bestMatch;
            count++;
        }

        return count > 0 ? similarity / count : 0;
    },

    normalizePalette: function(colors) {
        return colors.map(color => ({
            rgb: typeof color === 'string' ? color : color.rgb,
            prominence: color.prominence || color.score || 1
        }));
    },

    calculateColorMatch: function(color1, color2) {
        const rgb1 = this.parseRgb(color1.rgb);
        const rgb2 = this.parseRgb(color2.rgb);

        const distance = Math.sqrt(
            Math.pow(rgb1[0] - rgb2[0], 2) +
            Math.pow(rgb1[1] - rgb2[1], 2) +
            Math.pow(rgb1[2] - rgb2[2], 2)
        );

        return Math.max(0, 1 - distance / 441.67);
    },

    parseRgb: function(color) {
        if (typeof color === 'string') {
            const matches = color.match(/\d+/g);
            return matches ? matches.map(Number) : [0, 0, 0];
        }
        return [color.r || 0, color.g || 0, color.b || 0];
    }
};

// Add to searchUtils.js
async function enhancedImageSearch(query, options = {}) {
  const dropboxFiles = await dropboxManager.fetchDropboxEntries('');
  const imageFiles = dropboxFiles.result.entries.filter(entry =>
    ['.jpg', '.jpeg', '.png', '.gif', '.webp'].some(ext => 
      entry.path_lower.endsWith(ext)
    )
  );

  const results = [];
  for (const file of imageFiles) {
    const analysis = await imageProcessor.processImage({
      path: file.path_lower,
      metadata: {
        originalPath: file.path_lower,
        name: file.name,
        timestamp: file.server_modified
      }
    });
    
    if (analysis?.tattooFeatures) {
      results.push({
        path: file.path_lower,
        analysis,
        metadata: {
          name: file.name,
          modified: file.server_modified,
          size: file.size
        }
      });
    }
  }
  
  return results;
}

module.exports = searchUtils;
