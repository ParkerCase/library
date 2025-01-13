// colorAnalysis.js - Enhanced Production Version
const sharp = require('sharp');
const _ = require('lodash');
const mathUtils = require('./mathUtils');
const monitor = require('./monitoring');
const { ValidationError, ProcessingError } = require('./errorHandler');

class ColorAnalysis {
    // Constants for color space conversions

static COLOR_QUANT_LEVELS = {
        RGB: 64,
        LAB: 32,
        HSV: 48
    };

    static GABOR_ORIENTATIONS = [0, 30, 60, 90, 120, 150];
    static GABOR_FREQUENCIES = [0.1, 0.2, 0.3, 0.4];
    static GABOR_SIGMA = 3.0;
    static GABOR_GAMMA = 0.5;


    static XYZ_RGB_MATRIX = [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ];


    // D65 reference white point
    static LAB_REFERENCE = {
        X: 95.047,
        Y: 100.000,
        Z: 108.883
    };


constructor() {
        this.monitor = require('./monitoring');
    }

async analyzeImage(buffer) {
        const opId = this.monitor.startOperation('colorAnalysis');
        
        try {
            if (!Buffer.isBuffer(buffer)) {
                throw new ValidationError('Invalid input: Buffer expected');
            }

            const rawPixels = await ColorAnalysis.extractRawPixels(buffer);

            // Handle missing COLOR_QUANT_LEVELS
            if (!this.constructor.COLOR_QUANT_LEVELS) {
                this.constructor.COLOR_QUANT_LEVELS = ColorAnalysis.COLOR_QUANT_LEVELS;
            }

            // Parallel analysis with error handling and fallbacks
            const [
                dominantColors,
                histograms,
                labAnalysis,
                coherenceAnalysis,
                spatialAnalysis
            ] = await Promise.all([
                this.extractDominantColors(rawPixels).catch(() => []),
                this.generateColorHistograms(rawPixels).catch(() => this.getDefaultHistograms()),
                this.analyzeLABColorSpace(rawPixels).catch(() => this.getDefaultLABAnalysis()),
                this.analyzeColorCoherence(rawPixels, rawPixels.meta).catch(() => this.getDefaultCoherenceAnalysis()),
                this.analyzeSpatialRelationships(rawPixels, rawPixels.meta).catch(() => this.getDefaultSpatialAnalysis())
            ]);

            return {
                version: '2.0',
                timestamp: Date.now(),
                dominant: dominantColors,
                histograms,
                labAnalysis,
                coherence: coherenceAnalysis,
                spatial: spatialAnalysis,
                metadata: {
                    processingTime: Date.now() - this.monitor.startTime
                }
            };

        } catch (error) {
            if (error instanceof ValidationError) {
                throw error;
            }
            throw new ProcessingError('Color analysis failed', { error });
        } finally {
            this.monitor.endOperation(opId);
        }
    }


async analyzeColorCoherence(rawPixels, meta) {
        try {
            // Analyze connected components with enhanced error handling
            const components = await this.findConnectedComponentsEnhanced(rawPixels, meta);
            
            // Analyze component characteristics
            const [vectors, distribution, metrics] = await Promise.all([
                this.generateEnhancedCoherenceVectors(components),
                this.analyzeComponentDistributionEnhanced(components),
                this.calculateEnhancedCoherenceMetrics(components)
            ]);

            // Calculate spatial relationships between components
            const spatialAnalysis = await this.analyzeSpatialRelationships(components);

            return {
                vectors,
                distribution,
                metrics,
                spatialAnalysis,
                statistics: this.calculateComponentStatistics(components),
                quality: {
                    coherence: metrics.overallCoherence,
                    consistency: this.calculateColorConsistency(components),
                    complexity: this.calculateVisualComplexity(components)
                }
            };
        } catch (error) {
            console.error('Color coherence analysis failed:', error);
            return this.getDefaultCoherenceAnalysis();
        }
    }

    getDefaultCoherenceAnalysis() {
        return {
            vectors: [],
            distribution: {},
            metrics: {
                overallCoherence: 0,
                componentDistribution: {},
                spatialCoverage: 0
            },
            spatialAnalysis: {},
            statistics: {},
            quality: {
                coherence: 0,
                consistency: 0,
                complexity: 0
            }
        };
    }

 calculateColorDistance(color1, color2) {
        if (!color1 || !color2) return 0;
        return mathUtils.deltaE(color1, color2);
    }


    calculateClusterSpread(cluster) {
        if (!cluster || !cluster.points || !Array.isArray(cluster.points) || cluster.points.length === 0) {
            return {
                mean: 0,
                std: 0,
                max: 0
            };
        }

        const center = cluster.center || this.calculateClusterCenter(cluster.points);
        const distances = cluster.points.map(point => 
            this.calculateColorDistance(point, center)
        );

        return {
            mean: mathUtils.calculateMean(distances),
            std: Math.sqrt(mathUtils.calculateVariance(distances)),
            max: Math.max(...distances)
        };
    }

    // Make extractRawPixels a static method
static async extractRawPixels(buffer) {
        try {
            const { data, info } = await sharp(buffer)
                .raw()
                .ensureAlpha()
                .toBuffer({ resolveWithObject: true });

            return {
                data: new Uint8Array(data),
                info,
                meta: {
                    channels: info.channels,
                    width: info.width,
                    height: info.height
                }
            };
        } catch (error) {
            console.error('Raw pixel extraction failed:', error);
            throw new ProcessingError('Failed to extract raw pixels', { error });
        }
    }

async convertToLAB(rawPixels) {
        try {
            if (!rawPixels || !rawPixels.data) {
                return [];
            }

            const pixels = [];
            const { data, meta } = rawPixels;
            const stride = meta.channels;

            for (let i = 0; i < data.length; i += stride) {
                const rgb = {
                    r: Math.max(0, Math.min(255, data[i] || 0)),
                    g: Math.max(0, Math.min(255, data[i + 1] || 0)),
                    b: Math.max(0, Math.min(255, data[i + 2] || 0))
                };

                try {
                    const lab = mathUtils.rgbToLab(rgb);
                    pixels.push(lab);
                } catch (error) {
                    pixels.push({ l: 50, a: 0, b: 0 }); // Neutral gray as fallback
                }
            }

            return pixels;
        } catch (error) {
            console.error('Error in convertToLAB:', error);
            return [];
        }
    }


getDefaultLABAnalysis() {
        return {
            distribution: {
                channelStatistics: {
                    l: { mean: 50, median: 50, std: 0, skewness: 0, kurtosis: 0, range: 0 },
                    a: { mean: 0, median: 0, std: 0, skewness: 0, kurtosis: 0, range: 0 },
                    b: { mean: 0, median: 0, std: 0, skewness: 0, kurtosis: 0, range: 0 }
                },
                jointDistribution: [],
                density: new Map(),
                clusters: []
            },
            gamutCoverage: 0,
            perceptualSpread: 0,
            colorfulness: 0,
            statistics: {
                mean: { l: 50, a: 0, b: 0 },
                std: { l: 0, a: 0, b: 0 }
            },
            quality: {
                overall: 0,
                components: {
                    distribution: 0,
                    contrast: 0,
                    harmony: 0
                }
            }
        };
    }


// In ColorAnalysis class, change from static to instance method
async extractDominantColors(rawPixels, options = {}) {
    try {
        // Convert to LAB color space for perceptual clustering
        const labColors = await this.convertToLAB(rawPixels);
        
        // Enhanced clustering parameters
        const clusteringOptions = {
            bandWidth: options.bandWidth || 15,
            minClusterSize: options.minClusterSize || 
                Math.max(100, Math.floor(labColors.length * 0.01)),
            maxClusters: options.maxClusters || 8,
            convergenceThreshold: options.convergenceThreshold || 0.01
        };

        // Perform adaptive mean-shift clustering
        const clusters = await this.performAdaptiveClustering(
            labColors, 
            clusteringOptions
        );

        // Extract and enhance color information from clusters
        return clusters.map(cluster => ({
            lab: cluster.center,
            rgb: this.labToRGB(cluster.center),
            population: cluster.points.length / labColors.length,
            variance: this.calculateClusterVariance(cluster),
            distribution: this.analyzeColorDistribution(cluster),
            confidence: this.calculateClusterConfidence(cluster),
            metadata: {
                density: cluster.density,
                spread: cluster.spread,
                purity: this.calculateColorPurity(cluster)
            }
        })).sort((a, b) => b.population - a.population);
    } catch (error) {
        console.error('Dominant color extraction failed:', error);
        return [];
    }
}

// Also convert related methods to instance methods
async convertToLAB(rawPixels) {
    const pixels = [];
    const { data, meta } = rawPixels;
    const stride = meta.channels;

    for (let i = 0; i < data.length; i += stride) {
        const rgb = {
            r: Math.max(0, Math.min(255, data[i] || 0)),
            g: Math.max(0, Math.min(255, data[i + 1] || 0)),
            b: Math.max(0, Math.min(255, data[i + 2] || 0))
        };

        try {
            // Use mathUtils for the conversion instead of local implementation
            const lab = mathUtils.rgbToLab(rgb);
            pixels.push(lab);
        } catch (error) {
            // Use neutral gray as fallback
            pixels.push({ l: 50, a: 0, b: 0 });
        }
    }

    return pixels;
}

async performAdaptiveClustering(points, options) {
    const clusters = [];
    const processed = new Set();
    let bandWidth = options.bandWidth;

    while (clusters.length < options.maxClusters && bandWidth >= 5) {
        for (const point of points) {
            if (processed.has(point)) continue;

            const cluster = await this.growCluster(
                point,
                points,
                bandWidth,
                options,
                processed
            );

            if (cluster && cluster.points.length >= options.minClusterSize) {
                clusters.push(this.enhanceCluster(cluster));
                cluster.points.forEach(p => processed.add(p));
            }
        }

        // Adapt bandwidth if needed
        if (clusters.length < options.maxClusters / 2) {
            bandWidth *= 0.8;
        }
    }

    return this.refineClusters(clusters, options);
}

async growCluster(center, points, bandWidth, options, processed) {
    const cluster = {
        center: center,
        points: []
    };

    let oldCenter;
    let iterations = 0;
    const maxIterations = 50;

    do {
        oldCenter = { ...cluster.center };
        
        // Find points within bandwidth
        cluster.points = points.filter(p => 
            !processed.has(p) && 
            mathUtils.deltaE(p, cluster.center) <= bandWidth
        );

        if (cluster.points.length < options.minClusterSize) {
            return null;
        }

        // Update center
        const newCenter = this.calculateWeightedCenter(cluster.points);
        cluster.center = newCenter;

        iterations++;
    } while (
        mathUtils.deltaE(oldCenter, cluster.center) > options.convergenceThreshold && 
        iterations < maxIterations
    );

    return cluster;
}


calculateWeightedCenter(points) {
    const weights = points.map(p => this.calculatePointWeight(p));
    const weightSum = weights.reduce((a, b) => a + b, 0);

    return {
        l: points.reduce((sum, p, i) => sum + p.l * weights[i], 0) / weightSum,
        a: points.reduce((sum, p, i) => sum + p.a * weights[i], 0) / weightSum,
        b: points.reduce((sum, p, i) => sum + p.b * weights[i], 0) / weightSum
    };
}

calculatePointWeight(point) {
    const saturation = Math.sqrt(point.a * point.a + point.b * point.b);
    const lightness = point.l;
    
    const lightnessWeight = 1 - Math.abs(lightness - 50) / 50;
    const saturationWeight = saturation / 128;

    return lightnessWeight * saturationWeight;
}


enhanceCluster(cluster) {
    return {
        ...cluster,
        density: this.calculateClusterDensity(cluster),
        spread: this.calculateClusterSpread(cluster),
        boundary: this.calculateClusterBoundary(cluster),
        confidence: this.calculateClusterConfidence(cluster)
    };
}


    static refineClusters(clusters, options) {
        // Merge similar clusters
        let refined = this.mergeSimilarClusters(clusters, options.convergenceThreshold);

        // Split clusters that are too large or diverse
        refined = this.splitLargeClusters(refined, options);

        // Final adjustment of cluster centers
        return refined.map(cluster => ({
            ...cluster,
            center: this.optimizeClusterCenter(cluster)
        }));
    }

    static mergeSimilarClusters(clusters, threshold) {
        const merged = [...clusters];
        let didMerge;

        do {
            didMerge = false;
            for (let i = 0; i < merged.length; i++) {
                for (let j = i + 1; j < merged.length; j++) {
                    if (this.shouldMergeClusters(merged[i], merged[j], threshold)) {
                        merged[i] = this.mergeClusters(merged[i], merged[j]);
                        merged.splice(j, 1);
                        didMerge = true;
                        break;
                    }
                }
                if (didMerge) break;
            }
        } while (didMerge);

        return merged;
    }

    static shouldMergeClusters(cluster1, cluster2, threshold) {
        const colorDiff = mathUtils.deltaE(cluster1.center, cluster2.center);
        const densityRatio = Math.min(
            cluster1.density / cluster2.density,
            cluster2.density / cluster1.density
        );

        return colorDiff < threshold * 2 && densityRatio > 0.3;
    }

    static mergeClusters(cluster1, cluster2) {
        const totalPoints = cluster1.points.length + cluster2.points.length;
        const weight1 = cluster1.points.length / totalPoints;
        const weight2 = cluster2.points.length / totalPoints;

        return {
            center: {
                l: cluster1.center.l * weight1 + cluster2.center.l * weight2,
                a: cluster1.center.a * weight1 + cluster2.center.a * weight2,
                b: cluster1.center.b * weight1 + cluster2.center.b * weight2
            },
            points: [...cluster1.points, ...cluster2.points],
            density: (cluster1.density * weight1 + cluster2.density * weight2),
            spread: Math.max(cluster1.spread, cluster2.spread),
            confidence: Math.min(cluster1.confidence, cluster2.confidence)
        };
    }

    static optimizeClusterCenter(cluster) {
        // Use median values for better robustness
        const sortedL = _.sortBy(cluster.points, 'l');
        const sortedA = _.sortBy(cluster.points, 'a');
        const sortedB = _.sortBy(cluster.points, 'b');
        
        const mid = Math.floor(cluster.points.length / 2);
        
        return {
            l: sortedL[mid].l,
            a: sortedA[mid].a,
            b: sortedB[mid].b
        };
    }

    static calculateClusterVariance(cluster) {
        if (!cluster.points.length) return 0;

        const distances = cluster.points.map(point =>
            mathUtils.deltaE(point, cluster.center)
        );

        return {
            mean: _.mean(distances),
            std: Math.sqrt(_.variance(distances)),
            max: _.max(distances)
        };
    }



calculateClusterDensity(cluster) {
    if (!cluster.points.length) return 0;
    const volume = this.calculateClusterVolume(cluster);
    return cluster.points.length / volume;
}

calculateClusterVolume(cluster) {
    const points = cluster.points;
    if (!points.length) return 0;

    const bounds = points.reduce((acc, point) => ({
        minL: Math.min(acc.minL, point.l),
        maxL: Math.max(acc.maxL, point.l),
        minA: Math.min(acc.minA, point.a),
        maxA: Math.max(acc.maxA, point.a),
        minB: Math.min(acc.minB, point.b),
        maxB: Math.max(acc.maxB, point.b)
    }), {
        minL: Infinity,
        maxL: -Infinity,
        minA: Infinity,
        maxA: -Infinity,
        minB: Infinity,
        maxB: -Infinity
    });

    return (bounds.maxL - bounds.minL) * 
           (bounds.maxA - bounds.minA) * 
           (bounds.maxB - bounds.minB);
}

calculateClusterSpread(cluster) {
    if (!cluster.points.length) return 0;
    
    const center = cluster.center;
    const distances = cluster.points.map(point => 
        mathUtils.deltaE(point, center)
    );
    
    return {
        mean: mathUtils.calculateMean(distances),
        std: Math.sqrt(mathUtils.calculateVariance(distances)),
        max: Math.max(...distances)
    };
}

calculateClusterBoundary(cluster) {
    return {
        points: this.findBoundaryPoints(cluster.points),
        convexHull: this.computeConvexHull(cluster.points)
    };
}

calculateClusterConfidence(cluster) {
    const density = this.calculateClusterDensity(cluster);
    const spread = this.calculateClusterSpread(cluster);
    const separation = this.calculateClusterSeparation(cluster);

    return density * 0.4 + (1 - spread.mean / 100) * 0.3 + separation * 0.3;
}

static calculateClusterSeparation(cluster) {
        if (!global.clusters || !global.clusters.length) return 1;

        // Calculate minimum distance to other clusters
        const otherClusters = global.clusters.filter(c => c !== cluster);
        if (!otherClusters.length) return 1;

        const minDistance = Math.min(...otherClusters.map(other => 
            mathUtils.deltaE(cluster.center, other.center)
        ));

        // Normalize to [0,1] range with sigmoid function
        return 1 / (1 + Math.exp(-0.1 * (minDistance - 20)));
    }

 async generateColorHistograms(rawPixels) {
        if (!rawPixels || !rawPixels.data) {
            return this.getDefaultHistograms();
        }

        try {
            const [rgbHist, hsvHist, labHist] = await Promise.all([
                this.generateRGBHistogram(rawPixels),
                this.generateHSVHistogram(rawPixels),
                this.generateLABHistogram(rawPixels)
            ]);

            return {
                rgb: rgbHist,
                hsv: hsvHist,
                lab: labHist,
                statistics: this.calculateHistogramStatistics({
                    rgb: rgbHist,
                    hsv: hsvHist,
                    lab: labHist
                })
            };
        } catch (error) {
            console.error('Histogram generation failed:', error);
            return this.getDefaultHistograms();
        }
    }



generateRGBHistogram(rawPixels) {
        if (!this.constructor.COLOR_QUANT_LEVELS || !this.constructor.COLOR_QUANT_LEVELS.RGB) {
            this.constructor.COLOR_QUANT_LEVELS = ColorAnalysis.COLOR_QUANT_LEVELS;
        }

        const binCount = this.constructor.COLOR_QUANT_LEVELS.RGB;
        const histogram = new Float32Array(binCount * binCount * binCount).fill(0);
        
        if (!rawPixels || !rawPixels.data) {
            return histogram;
        }

        const data = rawPixels.data;
        for (let i = 0; i < data.length; i += 4) {
            const r = Math.floor(data[i] * (binCount - 1) / 255);
            const g = Math.floor(data[i + 1] * (binCount - 1) / 255);
            const b = Math.floor(data[i + 2] * (binCount - 1) / 255);
            
            const index = (r * binCount * binCount) + (g * binCount) + b;
            if (index >= 0 && index < histogram.length) {
                histogram[index]++;
            }
        }

        return this.normalizeHistogram(histogram);
    }


generateHSVHistogram(rawPixels) {
    const { data } = rawPixels;
    const binCount = this.COLOR_QUANT_LEVELS.HSV;
    const histogram = new Float32Array(binCount * binCount * binCount).fill(0);
    
    for (let i = 0; i < data.length; i += 4) {
        const hsv = this.rgbToHSV(data[i], data[i + 1], data[i + 2]);
        
        const h = Math.floor(hsv.h * (binCount - 1));
        const s = Math.floor(hsv.s * (binCount - 1));
        const v = Math.floor(hsv.v * (binCount - 1));
        
        const index = (h * binCount * binCount) + (s * binCount) + v;
        histogram[index]++;
    }

    return this.normalizeHistogram(histogram);
}


generateLABHistogram(rawPixels) {
    const { data } = rawPixels;
    const binCount = this.COLOR_QUANT_LEVELS.LAB;
    const histogram = new Float32Array(binCount * binCount * binCount).fill(0);
    
    for (let i = 0; i < data.length; i += 4) {
        const rgb = {
            r: data[i],
            g: data[i + 1],
            b: data[i + 2]
        };
        
        const lab = mathUtils.rgbToLab(rgb);
        
        const l = Math.floor((lab.l / 100) * (binCount - 1));
        const a = Math.floor(((lab.a + 128) / 255) * (binCount - 1));
        const b = Math.floor(((lab.b + 128) / 255) * (binCount - 1));
        
        const index = (l * binCount * binCount) + (a * binCount) + b;
        histogram[index]++;
    }

    return this.normalizeHistogram(histogram);
}

generateJointHistogram(rawPixels, colorSpace) {
    const { data } = rawPixels;
    const binCount = Math.floor(Math.sqrt(this.COLOR_QUANT_LEVELS[colorSpace.toUpperCase()]));
    const jointHist = Array(3).fill().map(() => 
        Array(binCount).fill().map(() => 
            new Float32Array(binCount).fill(0)
        )
    );

    for (let i = 0; i < data.length; i += 4) {
        let values;
        if (colorSpace === 'rgb') {
            values = [
                Math.floor(data[i] * (binCount - 1) / 255),
                Math.floor(data[i + 1] * (binCount - 1) / 255),
                Math.floor(data[i + 2] * (binCount - 1) / 255)
            ];
        } else {
            const hsv = this.rgbToHSV(data[i], data[i + 1], data[i + 2]);
            values = [
                Math.floor(hsv.h * (binCount - 1)),
                Math.floor(hsv.s * (binCount - 1)),
                Math.floor(hsv.v * (binCount - 1))
            ];
        }

        jointHist[0][values[0]][values[1]]++;
        jointHist[1][values[1]][values[2]]++;
        jointHist[2][values[0]][values[2]]++;
    }

    // Normalize each joint histogram
    return jointHist.map(plane => {
        const sum = plane.reduce((s, row) => 
            s + row.reduce((rs, val) => rs + val, 0), 0
        );
        return plane.map(row => row.map(val => val / sum));
    });
}



calculateHistogramStatistics(histograms) {
    return {
        rgb: this.calculateHistogramMoments(histograms.rgb),
        hsv: this.calculateHistogramMoments(histograms.hsv),
        lab: this.calculateHistogramMoments(histograms.lab),
        joint: this.calculateJointHistogramStatistics(histograms)
    };
}

calculateHistogramMoments(histogram) {
    const mean = mathUtils.calculateMean(histogram);
    const variance = mathUtils.calculateVariance(histogram, mean);
    const std = Math.sqrt(variance);

    return {
        mean,
        variance,
        std,
        skewness: mathUtils.calculateSkewness(histogram, mean, std),
        kurtosis: mathUtils.calculateKurtosis(histogram, mean, std),
        entropy: this.calculateEntropy(histogram)
    };
}

calculateJointHistogramStatistics(histograms) {
    return {
        mutual_information: this.calculateMutualInformation(histograms),
        cross_correlation: this.calculateCrossCorrelation(histograms),
        joint_entropy: this.calculateJointEntropy(histograms)
    };
}


 normalizeHistogram(histogram) {
        const sum = histogram.reduce((a, b) => a + b, 0);
        return sum > 0 ? histogram.map(val => val / sum) : histogram;
    }


    static calculateMutualInformation(histograms) {
        const result = {};
        ['rgb', 'hsv'].forEach(space => {
            const joint = histograms[`joint${space.toUpperCase()}`];
            result[space] = joint.map(matrix => {
                const marginal1 = matrix.map(row => row.reduce((sum, val) => sum + val, 0));
                const marginal2 = matrix[0].map((_, j) => 
                    matrix.reduce((sum, row) => sum + row[j], 0)
                );

                let mi = 0;
                matrix.forEach((row, i) => {
                    row.forEach((val, j) => {
                        if (val > 0) {
                            mi += val * Math.log2(val / (marginal1[i] * marginal2[j]));
                        }
                    });
                });

                return mi;
            });
        });
        return result;
    }

    static calculateCrossCorrelation(histograms) {
        const result = {};
        ['rgb', 'hsv'].forEach(space => {
            const joint = histograms[`joint${space.toUpperCase()}`];
            result[space] = joint.map(matrix => {
                const flatMatrix = matrix.flat();
                const mean = mathUtils.calculateMean(flatMatrix);
                const std = Math.sqrt(mathUtils.calculateVariance(flatMatrix, mean));

                let correlation = 0;
                matrix.forEach((row, i) => {
                    row.forEach((val, j) => {
                        correlation += (i - matrix.length/2) * 
                                     (j - row.length/2) * val;
                    });
                });

                return correlation / (matrix.length * matrix[0].length * std * std);
            });
        });
        return result;
    }

    static calculateJointEntropy(histograms) {
        const result = {};
        ['rgb', 'hsv'].forEach(space => {
            const joint = histograms[`joint${space.toUpperCase()}`];
            result[space] = joint.map(matrix => {
                let entropy = 0;
                matrix.forEach(row => {
                    row.forEach(val => {
                        if (val > 0) {
                            entropy -= val * Math.log2(val);
                        }
                    });
                });
                return entropy;
            });
        });
        return result;
    }

    static calculateEntropy(histogram) {
        return -histogram.reduce((sum, p) => {
            return sum + (p > 0 ? p * Math.log2(p) : 0);
        }, 0);
    }

    static rgbToHSV(r, g, b) {
        r /= 255;
        g /= 255;
        b /= 255;

        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const delta = max - min;

        let h = 0;
        if (delta !== 0) {
            if (max === r) {
                h = ((g - b) / delta) % 6;
            } else if (max === g) {
                h = (b - r) / delta + 2;
            } else {
                h = (r - g) / delta + 4;
            }
            h *= 60;
            if (h < 0) h += 360;
        }

        const s = max === 0 ? 0 : delta / max;
        const v = max;

        return {
            h: h / 360, // Normalize to [0,1]
            s,
            v
        };
    }

  async analyzeLABColorSpace(rawPixels) {
        try {
            if (!rawPixels || !rawPixels.data) {
                return this.getDefaultLABAnalysis();
            }

            // Convert to LAB colors
            const labColors = await this.convertToLAB(rawPixels);
            if (!Array.isArray(labColors)) {
                return this.getDefaultLABAnalysis();
            }

            // Analyze distribution in LAB space
            const distribution = this.analyzeLABDistribution(labColors);
            const coverage = this.calculateGamutCoverage(labColors);
            const spread = this.calculatePerceptualSpread(labColors);
            const colorfulness = this.calculateColorfulness(labColors);

            return {
                distribution,
                gamutCoverage: coverage,
                perceptualSpread: spread,
                colorfulness,
                statistics: this.calculateLABStatistics(labColors),
                quality: await this.assessLABQuality(labColors)
            };
        } catch (error) {
            console.error('LAB color space analysis failed:', error);
            return this.getDefaultLABAnalysis();
        }
    }



calculateMedian(values) {
    if (!values.length) return 0;
    const sorted = Array.from(values).sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? 
        (sorted[mid - 1] + sorted[mid]) / 2 : 
        sorted[mid];
}

calculateRange(values) {
    if (!values.length) return 0;
    return Math.max(...values) - Math.min(...values);
}

calculatePerceptualContrast(labColors) {
    if (!labColors || labColors.length < 2) return 0;
    
    const contrasts = [];
    for (let i = 0; i < labColors.length - 1; i++) {
        for (let j = i + 1; j < labColors.length; j++) {
            contrasts.push(mathUtils.deltaE(labColors[i], labColors[j]));
        }
    }
    
    return mathUtils.calculateMean(contrasts) / 100;
}

calculateColorVariation(labColors) {
    if (!labColors || !labColors.length) return 0;
    
    const variations = {
        lightness: this.calculateChannelVariation(labColors.map(c => c.l)),
        chroma: this.calculateChannelVariation(labColors.map(c => 
            Math.sqrt(c.a * c.a + c.b * c.b)
        )),
        hue: this.calculateHueVariation(labColors)
    };
    
    return (variations.lightness + variations.chroma + variations.hue) / 3;
}


analyzeLABDistribution(labColors) {
    // Calculate distributions for each channel
    const channels = {
        l: labColors.map(color => color.l),
        a: labColors.map(color => color.a),
        b: labColors.map(color => color.b)
    };

    // Calculate statistics for each channel
    const statistics = {};
    for (const [channel, values] of Object.entries(channels)) {
        statistics[channel] = {
            mean: mathUtils.calculateMean(values),
            median: this.calculateMedian(values),
            std: Math.sqrt(mathUtils.calculateVariance(values)),
            skewness: mathUtils.calculateSkewness(values),
            kurtosis: mathUtils.calculateKurtosis(values),
            range: this.calculateRange(values)
        };
    }

    return {
        channelStatistics: statistics,
        jointDistribution: this.calculateJointLABDistribution(labColors),
        density: this.calculateDensityEstimation(labColors),
        clusters: this.identifyLABClusters(labColors)
    };
}

    static calculateJointLABDistribution(labColors) {
        const binCount = 16; // Number of bins for each dimension
        const distribution = Array(binCount).fill().map(() => 
            Array(binCount).fill().map(() => 
                Array(binCount).fill(0)
            )
        );

        // Calculate bin indices for each color
        labColors.forEach(color => {
            const lBin = Math.floor((color.l / 100) * (binCount - 1));
            const aBin = Math.floor(((color.a + 128) / 256) * (binCount - 1));
            const bBin = Math.floor(((color.b + 128) / 256) * (binCount - 1));
            
            if (lBin >= 0 && lBin < binCount &&
                aBin >= 0 && aBin < binCount &&
                bBin >= 0 && bBin < binCount) {
                distribution[lBin][aBin][bBin]++;
            }
        });

        // Normalize distribution
        const total = labColors.length;
        return distribution.map(plane => 
            plane.map(row => 
                row.map(val => val / total)
            )
        );
    }

    static calculateDensityEstimation(labColors) {
        // Kernel Density Estimation with adaptive bandwidth
        const bandwidth = this.calculateAdaptiveBandwidth(labColors);
        const densityMap = new Map();

        labColors.forEach(color => {
            const key = this.quantizeLABColor(color);
            if (!densityMap.has(key)) {
                densityMap.set(key, 0);
            }
            densityMap.set(key, densityMap.get(key) + 1);
        });

        // Normalize density values
        const maxDensity = Math.max(...densityMap.values());
        const normalizedDensity = new Map();
        densityMap.forEach((value, key) => {
            normalizedDensity.set(key, value / maxDensity);
        });

        return {
            density: Object.fromEntries(normalizedDensity),
            bandwidth,
            statistics: this.calculateDensityStatistics(normalizedDensity)
        };
    }

    static calculateAdaptiveBandwidth(labColors) {
        // Scott's rule for bandwidth selection
        const n = labColors.length;
        const dim = 3; // LAB space is 3-dimensional
        
        const standardDeviations = {
            l: Math.sqrt(mathUtils.calculateVariance(labColors.map(c => c.l))),
            a: Math.sqrt(mathUtils.calculateVariance(labColors.map(c => c.a))),
            b: Math.sqrt(mathUtils.calculateVariance(labColors.map(c => c.b)))
        };

        return {
            l: standardDeviations.l * Math.pow(n, -1/(dim + 4)),
            a: standardDeviations.a * Math.pow(n, -1/(dim + 4)),
            b: standardDeviations.b * Math.pow(n, -1/(dim + 4))
        };
    }

    static quantizeLABColor(color) {
        // Quantize LAB color to reduce number of unique values
        const quantizationLevels = {
            l: 20, // Levels for L channel
            a: 15, // Levels for a channel
            b: 15  // Levels for b channel
        };

        const quantized = {
            l: Math.floor(color.l / 100 * quantizationLevels.l),
            a: Math.floor((color.a + 128) / 256 * quantizationLevels.a),
            b: Math.floor((color.b + 128) / 256 * quantizationLevels.b)
        };

        return `${quantized.l},${quantized.a},${quantized.b}`;
    }

    static calculateGamutCoverage(labColors) {
        // Calculate coverage in different regions of LAB color space
        const regions = this.defineLabRegions();
        const coverage = new Map();

        regions.forEach(region => {
            const colorsInRegion = labColors.filter(color => 
                this.isColorInRegion(color, region)
            );
            coverage.set(region.name, colorsInRegion.length / labColors.length);
        });

        return {
            regionalCoverage: Object.fromEntries(coverage),
            totalCoverage: this.calculateTotalGamutCoverage(labColors),
            statistics: this.analyzeGamutCoverageStats(coverage)
        };
    }

    static defineLabRegions() {
        return [
            {
                name: 'light',
                bounds: { lMin: 50, lMax: 100, aMin: -128, aMax: 128, bMin: -128, bMax: 128 }
            },
            {
                name: 'dark',
                bounds: { lMin: 0, lMax: 50, aMin: -128, aMax: 128, bMin: -128, bMax: 128 }
            },
            {
                name: 'vivid',
                bounds: { lMin: 25, lMax: 75, aMin: -128, aMax: 128, bMin: -128, bMax: 128 },
                condition: (color) => Math.sqrt(color.a * color.a + color.b * color.b) > 60
            },
            {
                name: 'neutral',
                bounds: { lMin: 0, lMax: 100, aMin: -30, aMax: 30, bMin: -30, bMax: 30 }
            },
            {
                name: 'warm',
                bounds: { lMin: 0, lMax: 100, aMin: 0, aMax: 128, bMin: 0, bMax: 128 }
            },
            {
                name: 'cool',
                bounds: { lMin: 0, lMax: 100, aMin: -128, aMax: 0, bMin: -128, bMax: 0 }
            }
        ];
    }

    static isColorInRegion(color, region) {
        const { bounds, condition } = region;
        const inBounds = color.l >= bounds.lMin && color.l <= bounds.lMax &&
                        color.a >= bounds.aMin && color.a <= bounds.aMax &&
                        color.b >= bounds.bMin && color.b <= bounds.bMax;

        return inBounds && (!condition || condition(color));
    }

    static calculateTotalGamutCoverage(labColors) {
        // Calculate volume of color space occupied
        const convexHull = this.calculateConvexHull(labColors);
        const volume = this.calculateConvexHullVolume(convexHull);
        
        // Compare to theoretical maximum volume
        const maxVolume = 100 * 256 * 256; // Approximate LAB color space volume
        return volume / maxVolume;
    }

    static calculateConvexHull(points) {
        // Implementation of QuickHull algorithm for 3D points
        // This is a simplified version for demonstration
        // In production, use a robust computational geometry library
        return points;
    }

    static calculateConvexHullVolume(hull) {
        // Simplified volume calculation
        // In production, use proper geometric calculations
        const bounds = this.calculateBounds(hull);
        return Math.abs(
            (bounds.maxL - bounds.minL) *
            (bounds.maxA - bounds.minA) *
            (bounds.maxB - bounds.minB)
        );
    }

    static calculatePerceptualSpread(labColors) {
        // Calculate perceptual distance between colors
        const distances = [];
        for (let i = 0; i < labColors.length - 1; i++) {
            for (let j = i + 1; j < labColors.length; j++) {
                distances.push(mathUtils.deltaE(labColors[i], labColors[j]));
            }
        }

        return {
            mean: mathUtils.calculateMean(distances),
            median: this.calculateMedian(distances),
            std: Math.sqrt(mathUtils.calculateVariance(distances)),
            max: Math.max(...distances),
            distribution: this.calculateDistanceDistribution(distances)
        };
    }

static calculateColorfulness(labColors) {
        // Enhanced colorfulness calculation using multiple metrics
        const [chroma, saturation] = this.calculateChromaAndSaturation(labColors);
        const colorVariety = this.calculateColorVariety(labColors);
        const spread = this.calculateSpread(labColors);

        // Combine metrics with weighted importance
        return {
            value: (chroma * 0.3 + saturation * 0.3 + colorVariety * 0.2 + spread * 0.2),
            components: {
                chroma,
                saturation,
                colorVariety,
                spread
            },
            distribution: this.calculateColorfulnessDistribution(labColors)
        };
    }

    static calculateChromaAndSaturation(labColors) {
        const chromas = labColors.map(color => 
            Math.sqrt(color.a * color.a + color.b * color.b)
        );

        const saturations = labColors.map(color => {
            const chroma = Math.sqrt(color.a * color.a + color.b * color.b);
            return chroma / (Math.sqrt(chroma * chroma + color.l * color.l) || 1);
        });

        return [
            mathUtils.calculateMean(chromas) / 128, // Normalize to [0,1]
            mathUtils.calculateMean(saturations)
        ];
    }

    static calculateColorVariety(labColors) {
        // Calculate unique colors after quantization
        const quantized = new Set();
        labColors.forEach(color => {
            quantized.add(this.quantizeLABColor(color));
        });

        // Normalize by theoretical maximum
        const maxPossibleColors = 20 * 15 * 15; // Based on quantization levels
        return Math.min(quantized.size / maxPossibleColors, 1);
    }

    static calculateSpread(labColors) {
        // Calculate spread in LAB space
        const bounds = this.calculateBounds(labColors);
        const maxPossibleSpread = Math.sqrt(100 * 100 + 256 * 256 + 256 * 256);
        
        const spread = Math.sqrt(
            Math.pow(bounds.maxL - bounds.minL, 2) +
            Math.pow(bounds.maxA - bounds.minA, 2) +
            Math.pow(bounds.maxB - bounds.minB, 2)
        );

        return spread / maxPossibleSpread;
    }

    static async analyzeColorCoherence(rawPixels, metadata) {
        try {
            // Enhanced connected component analysis
            const components = await this.findConnectedComponentsEnhanced(rawPixels, metadata);
            
            // Analyze component characteristics
            const [vectors, distribution, metrics] = await Promise.all([
                this.generateEnhancedCoherenceVectors(components),
                this.analyzeComponentDistributionEnhanced(components),
                this.calculateEnhancedCoherenceMetrics(components)
            ]);

            // Calculate spatial relationships between components
            const spatialAnalysis = await this.analyzeSpatialRelationships(components);

            return {
                vectors,
                distribution,
                metrics,
                spatialAnalysis,
                statistics: this.calculateComponentStatistics(components),
                quality: {
                    coherence: metrics.overallCoherence,
                    consistency: this.calculateColorConsistency(components),
                    complexity: this.calculateVisualComplexity(components)
                }
            };
        } catch (error) {
            console.error('Color coherence analysis failed:', error);
            return this.getDefaultCoherenceAnalysis();
        }
    }

    static async findConnectedComponentsEnhanced(rawPixels, metadata) {
        const { width, height } = metadata;
        const visited = new Set();
        const components = [];
        const threshold = this.calculateAdaptiveThreshold(rawPixels);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const idx = (y * width + x) * 4;
                if (!visited.has(idx)) {
                    const component = await this.growComponentEnhanced(
                        rawPixels, x, y, width, height, visited, threshold
                    );
                    if (this.isSignificantComponent(component)) {
                        components.push(this.enhanceComponent(component));
                    }
                }
            }
        }

        return this.postProcessComponents(components);
    }

    static calculateAdaptiveThreshold(rawPixels) {
        // Calculate noise level and color variation
        const noiseLevel = this.estimateImageNoise(rawPixels);
        const colorVariation = this.calculateColorVariation(rawPixels);

        // Adjust threshold based on image characteristics
        return {
            colorDifference: Math.max(25, Math.min(45, 30 + noiseLevel * 10)),
            spatialDistance: Math.max(1, Math.min(3, 2 + colorVariation))
        };
    }

    static async growComponentEnhanced(rawPixels, startX, startY, width, height, visited, threshold) {
        const component = this.initializeComponent(startX, startY);
        const stack = [[startX, startY]];
        const targetColor = this.getPixelColor(rawPixels, startX, startY, width);
        const labTarget = mathUtils.rgbToLab(targetColor);

        while (stack.length > 0) {
            const [x, y] = stack.pop();
            const idx = (y * width + x) * 4;

            if (visited.has(idx)) continue;
            visited.add(idx);

            const currentColor = this.getPixelColor(rawPixels, x, y, width);
            const labCurrent = mathUtils.rgbToLab(currentColor);

            if (this.isColorSimilarEnhanced(labTarget, labCurrent, threshold)) {
                this.updateComponent(component, x, y, currentColor);
                this.addNeighborsToStack(stack, x, y, width, height, visited);
            }
        }

        return this.finalizeComponent(component);
    }

    static isColorSimilarEnhanced(lab1, lab2, threshold) {
        // Enhanced color similarity using deltaE and additional metrics
        const deltaE = mathUtils.deltaE(lab1, lab2);
        if (deltaE > threshold.colorDifference) return false;

        // Check additional color characteristics
        const chromaDiff = Math.abs(
            Math.sqrt(lab1.a * lab1.a + lab1.b * lab1.b) -
            Math.sqrt(lab2.a * lab2.a + lab2.b * lab2.b)
        );

        const hueDiff = Math.abs(
            Math.atan2(lab1.b, lab1.a) -
            Math.atan2(lab2.b, lab2.a)
        );

        return chromaDiff < threshold.colorDifference / 2 &&
               hueDiff < Math.PI / 4;
    }

    static enhanceComponent(component) {
        return {
            ...component,
            metrics: {
                compactness: this.calculateCompactness(component),
                regularity: this.calculateRegularity(component),
                boundary: this.analyzeBoundary(component),
                texture: this.analyzeComponentTexture(component)
            },
            color: {
                ...component.color,
                lab: mathUtils.rgbToLab(component.color),
                variance: this.calculateColorVariance(component),
                distribution: this.analyzeColorDistribution(component)
            },
            shape: {
                orientation: this.calculateOrientation(component),
                elongation: this.calculateElongation(component),
                symmetry: this.calculateSymmetry(component),
                complexity: this.calculateShapeComplexity(component)
            }
        };
    }

    static calculateCompactness(component) {
        const perimeter = this.calculatePerimeter(component.pixels);
        const area = component.pixels.length;
        return (4 * Math.PI * area) / (perimeter * perimeter);
    }

    static calculateRegularity(component) {
        // Analyze pixel distribution regularity
        const centroid = this.calculateCentroid(component.pixels);
        const distances = component.pixels.map(([x, y]) =>
            Math.sqrt(Math.pow(x - centroid.x, 2) + Math.pow(y - centroid.y, 2))
        );

        return {
            mean: mathUtils.calculateMean(distances),
            std: Math.sqrt(mathUtils.calculateVariance(distances)),
            uniformity: this.calculateUniformity(distances)
        };
    }

    static analyzeBoundary(component) {
        const boundary = this.extractBoundary(component.pixels);
        const curvature = this.calculateBoundaryCurvature(boundary);
        const roughness = this.calculateBoundaryRoughness(boundary);

        return {
            points: boundary,
            length: boundary.length,
            curvature,
            roughness,
            features: this.extractBoundaryFeatures(boundary)
        };
    }

    static extractBoundary(pixels) {
        const boundary = [];
        const pixelSet = new Set(pixels.map(([x, y]) => `${x},${y}`));

        pixels.forEach(([x, y]) => {
            // Check if pixel is on boundary
            const neighbors = [
                [x-1, y], [x+1, y],
                [x, y-1], [x, y+1]
            ];

            if (neighbors.some(([nx, ny]) => !pixelSet.has(`${nx},${ny}`))) {
                boundary.push([x, y]);
            }
        });

        return this.orderBoundaryPoints(boundary);
    }

    static calculateBoundaryCurvature(boundary) {
        const curvatures = [];
        const windowSize = 5;

        for (let i = 0; i < boundary.length; i++) {
            const points = [];
            for (let j = -windowSize; j <= windowSize; j++) {
                const idx = (i + j + boundary.length) % boundary.length;
                points.push(boundary[idx]);
            }
            curvatures.push(this.calculateLocalCurvature(points));
        }

        return {
            values: curvatures,
            mean: mathUtils.calculateMean(curvatures),
            max: Math.max(...curvatures),
            distribution: this.analyzeDistribution(curvatures)
        };
    }

    static calculateLocalCurvature(points) {
        // Fit circle to points and calculate curvature
        // This is a simplified version; in production use more robust methods
        const center = points[Math.floor(points.length / 2)];
        const distances = points.map(([x, y]) =>
            Math.sqrt(Math.pow(x - center[0], 2) + Math.pow(y - center[1], 2))
        );
        
        return 1 / mathUtils.calculateMean(distances);
    }

static async analyzeColorHarmonies(dominantColors) {
        try {
            // Convert colors to HSV for harmony analysis
            const hsvColors = dominantColors.map(color => ({
                ...this.rgbToHSV(color.rgb.r, color.rgb.g, color.rgb.b),
                weight: color.population
            }));

            // Analyze different harmony types
            const harmonies = {
                complementary: this.findComplementaryHarmonies(hsvColors),
                analogous: this.findAnalogousHarmonies(hsvColors),
                triadic: this.findTriadicHarmonies(hsvColors),
                tetradic: this.findTetradicHarmonies(hsvColors),
                monochromatic: this.findMonochromaticHarmonies(hsvColors)
            };

            // Calculate overall harmony score
            const harmonyScore = this.calculateOverallHarmony(harmonies);

            // Analyze color balance
            const balance = this.analyzeColorBalance(hsvColors);

            return {
                harmonies,
                score: harmonyScore,
                balance,
                relationships: this.analyzeColorRelationships(hsvColors),
                recommendations: this.generateHarmonyRecommendations(harmonies)
            };
        } catch (error) {
            console.error('Color harmony analysis failed:', error);
            return this.getDefaultHarmonyAnalysis();
        }
    }

    static findComplementaryHarmonies(colors) {
        const pairs = [];
        const threshold = 15; // degrees in hue

        for (let i = 0; i < colors.length - 1; i++) {
            for (let j = i + 1; j < colors.length; j++) {
                const hueDiff = Math.abs(colors[i].h - colors[j].h) * 360;
                if (Math.abs(hueDiff - 180) <= threshold) {
                    pairs.push({
                        colors: [colors[i], colors[j]],
                        strength: 1 - Math.abs(hueDiff - 180) / threshold,
                        weight: colors[i].weight * colors[j].weight
                    });
                }
            }
        }

        return {
            pairs,
            strength: this.calculateHarmonyStrength(pairs)
        };
    }

    static analyzeColorRelationships(colors) {
        const relationships = [];
        
        for (let i = 0; i < colors.length - 1; i++) {
            for (let j = i + 1; j < colors.length; j++) {
                relationships.push({
                    colors: [colors[i], colors[j]],
                    hueDifference: this.calculateHueDifference(colors[i], colors[j]),
                    saturationRatio: colors[i].s / colors[j].s,
                    valueDifference: Math.abs(colors[i].v - colors[j].v),
                    weight: colors[i].weight * colors[j].weight
                });
            }
        }

        return {
            pairs: relationships,
            statistics: this.calculateRelationshipStatistics(relationships)
        };
    }

    static calculateHueDifference(color1, color2) {
        let diff = Math.abs(color1.h - color2.h);
        return Math.min(diff, 1 - diff) * 360;
    }

    static calculateOverallHarmony(harmonies) {
        const weights = {
            complementary: 0.25,
            analogous: 0.2,
            triadic: 0.2,
            tetradic: 0.15,
            monochromatic: 0.2
        };

        return Object.entries(harmonies).reduce((score, [type, harmony]) => {
            return score + harmony.strength * weights[type];
        }, 0);
    }

    static analyzeColorBalance(colors) {
        // Calculate weighted centroid in HSV space
        const weightedCentroid = colors.reduce((acc, color) => ({
            h: acc.h + color.h * color.weight,
            s: acc.s + color.s * color.weight,
            v: acc.v + color.v * color.weight
        }), { h: 0, s: 0, v: 0 });

        const totalWeight = colors.reduce((sum, color) => sum + color.weight, 0);
        
        weightedCentroid.h /= totalWeight;
        weightedCentroid.s /= totalWeight;
        weightedCentroid.v /= totalWeight;

        return {
            centroid: weightedCentroid,
            distribution: this.analyzeColorDistribution(colors),
            balance: this.calculateBalanceMetrics(colors, weightedCentroid)
        };
    }

    static calculateBalanceMetrics(colors, centroid) {
        return {
            hueBalance: this.calculateHueBalance(colors),
            saturationBalance: this.calculateSaturationBalance(colors),
            valueBalance: this.calculateValueBalance(colors),
            spatialBalance: this.calculateSpatialBalance(colors)
        };
    }

    // Quality assessment methods
    static async assessColorQuality(rawPixels) {
        const startTime = Date.now();

        try {
            const [
                contrast,
                colorfulness,
                naturalness,
                complexity
            ] = await Promise.all([
                this.assessContrast(rawPixels),
                this.assessColorfulness(rawPixels),
                this.assessNaturalness(rawPixels),
                this.assessComplexity(rawPixels)
            ]);

            const quality = this.combineQualityMetrics({
                contrast,
                colorfulness,
                naturalness,
                complexity
            });

            return {
                quality,
                metrics: {
                    contrast,
                    colorfulness,
                    naturalness,
                    complexity
                },
                confidence: this.calculateQualityConfidence({
                    contrast,
                    colorfulness,
                    naturalness,
                    complexity
                }),
                processingTime: Date.now() - startTime
            };
        } catch (error) {
            console.error('Color quality assessment failed:', error);
            return this.getDefaultQualityAssessment();
        }
    }

    static combineQualityMetrics(metrics) {
        const weights = {
            contrast: 0.3,
            colorfulness: 0.25,
            naturalness: 0.25,
            complexity: 0.2
        };

        return Object.entries(metrics).reduce((quality, [metric, value]) => {
            return quality + value.score * weights[metric];
        }, 0);
    }

    // Default analysis getters
    static getDefaultAnalysis(component) {
        return {
            dominantColors: [],
            colorHistograms: this.getDefaultHistograms(),
            labAnalysis: this.getDefaultLABAnalysis(),
            colorCoherence: this.getDefaultCoherenceAnalysis(),
            spatialRelationships: this.getDefaultSpatialAnalysis()
        };
    }


getDefaultHistograms() {
        const rgb = new Float32Array(this.constructor.COLOR_QUANT_LEVELS.RGB ** 3).fill(1 / (this.constructor.COLOR_QUANT_LEVELS.RGB ** 3));
        const hsv = new Float32Array(this.constructor.COLOR_QUANT_LEVELS.HSV ** 3).fill(1 / (this.constructor.COLOR_QUANT_LEVELS.HSV ** 3));
        const lab = new Float32Array(this.constructor.COLOR_QUANT_LEVELS.LAB ** 3).fill(1 / (this.constructor.COLOR_QUANT_LEVELS.LAB ** 3));

        return {
            rgb,
            hsv,
            lab,
            statistics: {
                rgb: this.getDefaultHistogramStatistics(),
                hsv: this.getDefaultHistogramStatistics(),
                lab: this.getDefaultHistogramStatistics()
            }
        };
    }

    getDefaultHistogramStatistics() {
        return {
            mean: 0,
            variance: 0,
            std: 0,
            skewness: 0,
            kurtosis: 0,
            entropy: 0
        };
    }


identifyLABClusters(labColors) {
    const clusters = [];
    const processedPoints = new Set();
    const epsilon = 5; // Distance threshold
    const minPoints = Math.max(5, Math.floor(labColors.length * 0.01));

    for (const point of labColors) {
        if (processedPoints.has(point)) continue;

        const clusterPoints = this.regionQuery(point, labColors, epsilon);
        if (clusterPoints.length >= minPoints) {
            const cluster = {
                points: clusterPoints,
                center: this.calculateClusterCenter(clusterPoints),
                density: clusterPoints.length / this.calculateClusterVolume({ points: clusterPoints })
            };
            clusters.push(this.enhanceCluster(cluster));
            clusterPoints.forEach(p => processedPoints.add(p));
        }
    }

    return this.mergeSimilarClusters(clusters);
}

regionQuery(point, points, epsilon) {
    return points.filter(p => mathUtils.deltaE(point, p) <= epsilon);
}

calculateClusterCenter(points) {
    if (!points.length) return { l: 0, a: 0, b: 0 };
    
    return {
        l: points.reduce((sum, p) => sum + p.l, 0) / points.length,
        a: points.reduce((sum, p) => sum + p.a, 0) / points.length,
        b: points.reduce((sum, p) => sum + p.b, 0) / points.length
    };
}

calculateClusterSeparation(cluster) {
    if (!global.clusters || !global.clusters.length) return 1;

    const otherClusters = global.clusters.filter(c => c !== cluster);
    if (!otherClusters.length) return 1;

    const minDistance = Math.min(...otherClusters.map(other => 
        mathUtils.deltaE(cluster.center, other.center)
    ));

    return 1 / (1 + Math.exp(-0.1 * (minDistance - 20)));
}

mergeSimilarClusters(clusters) {
    let merged = [...clusters];
    let changed;

    do {
        changed = false;
        for (let i = 0; i < merged.length; i++) {
            for (let j = i + 1; j < merged.length; j++) {
                if (this.shouldMergeClusters(merged[i], merged[j])) {
                    merged[i] = this.mergeClusters(merged[i], merged[j]);
                    merged.splice(j, 1);
                    changed = true;
                    break;
                }
            }
            if (changed) break;
        }
    } while (changed);

    return merged;
}

shouldMergeClusters(cluster1, cluster2) {
    const colorDiff = mathUtils.deltaE(cluster1.center, cluster2.center);
    const densityRatio = Math.min(
        cluster1.density / cluster2.density,
        cluster2.density / cluster1.density
    );

    return colorDiff < 15 && densityRatio > 0.3;
}

mergeClusters(cluster1, cluster2) {
    const totalPoints = cluster1.points.length + cluster2.points.length;
    const weight1 = cluster1.points.length / totalPoints;
    const weight2 = cluster2.points.length / totalPoints;

    return this.enhanceCluster({
        points: [...cluster1.points, ...cluster2.points],
        center: {
            l: cluster1.center.l * weight1 + cluster2.center.l * weight2,
            a: cluster1.center.a * weight1 + cluster2.center.a * weight2,
            b: cluster1.center.b * weight1 + cluster2.center.b * weight2
        }
    });
}

calculateJointLABDistribution(labColors) {
    const binCount = 16;
    const distribution = Array(binCount).fill().map(() => 
        Array(binCount).fill().map(() => 
            Array(binCount).fill(0)
        )
    );

    labColors.forEach(color => {
        const lBin = Math.floor((color.l / 100) * (binCount - 1));
        const aBin = Math.floor(((color.a + 128) / 256) * (binCount - 1));
        const bBin = Math.floor(((color.b + 128) / 256) * (binCount - 1));
        
        if (lBin >= 0 && lBin < binCount &&
            aBin >= 0 && aBin < binCount &&
            bBin >= 0 && bBin < binCount) {
            distribution[lBin][aBin][bBin]++;
        }
    });

    const total = labColors.length;
    return distribution.map(plane => 
        plane.map(row => 
            row.map(val => val / total)
        )
    );
}

calculateDensityEstimation(labColors) {
    const bandwidth = this.calculateAdaptiveBandwidth(labColors);
    const densityMap = new Map();

    labColors.forEach(color => {
        const key = this.quantizeLABColor(color);
        densityMap.set(key, (densityMap.get(key) || 0) + 1);
    });

    const maxDensity = Math.max(...densityMap.values());
    const normalizedDensity = new Map();
    densityMap.forEach((value, key) => {
        normalizedDensity.set(key, value / maxDensity);
    });

    return {
        density: Object.fromEntries(normalizedDensity),
        bandwidth,
        statistics: this.calculateDensityStatistics(normalizedDensity)
    };
}

calculateAdaptiveBandwidth(labColors) {
    const n = labColors.length;
    const dim = 3; // LAB space is 3-dimensional
    
    const standardDeviations = {
        l: Math.sqrt(mathUtils.calculateVariance(labColors.map(c => c.l))),
        a: Math.sqrt(mathUtils.calculateVariance(labColors.map(c => c.a))),
        b: Math.sqrt(mathUtils.calculateVariance(labColors.map(c => c.b)))
    };

    return {
        l: standardDeviations.l * Math.pow(n, -1/(dim + 4)),
        a: standardDeviations.a * Math.pow(n, -1/(dim + 4)),
        b: standardDeviations.b * Math.pow(n, -1/(dim + 4))
    };
}

quantizeLABColor(color) {
    const quantizationLevels = {
        l: 20,
        a: 15,
        b: 15
    };

    const quantized = {
        l: Math.floor(color.l / 100 * quantizationLevels.l),
        a: Math.floor((color.a + 128) / 256 * quantizationLevels.a),
        b: Math.floor((color.b + 128) / 256 * quantizationLevels.b)
    };

    return `${quantized.l},${quantized.a},${quantized.b}`;
}

calculateDensityStatistics(density) {
    const values = Array.from(density.values());
    return {
        mean: mathUtils.calculateMean(values),
        std: Math.sqrt(mathUtils.calculateVariance(values)),
        entropy: this.calculateEntropy(values),
        peaks: this.findLocalMaxima(values)
    };
}

calculateEntropy(values) {
    const sum = values.reduce((a, b) => a + b, 0);
    if (sum === 0) return 0;

    return -values.reduce((entropy, value) => {
        const p = value / sum;
        return entropy + (p > 0 ? p * Math.log2(p) : 0);
    }, 0);
}

findLocalMaxima(values) {
    const peaks = [];
    for (let i = 1; i < values.length - 1; i++) {
        if (values[i] > values[i - 1] && values[i] > values[i + 1]) {
            peaks.push({
                index: i,
                value: values[i]
            });
        }
    }
    return peaks;
}

calculateGamutCoverage(labColors) {
    const regions = this.defineLabRegions();
    const coverage = new Map();

    regions.forEach(region => {
        const colorsInRegion = labColors.filter(color => 
            this.isColorInRegion(color, region)
        );
        coverage.set(region.name, colorsInRegion.length / labColors.length);
    });

    return {
        regionalCoverage: Object.fromEntries(coverage),
        totalCoverage: this.calculateTotalGamutCoverage(labColors),
        statistics: this.analyzeGamutCoverageStats(coverage)
    };
}

defineLabRegions() {
    return [
        {
            name: 'light',
            bounds: { lMin: 50, lMax: 100, aMin: -128, aMax: 128, bMin: -128, bMax: 128 }
        },
        {
            name: 'dark',
            bounds: { lMin: 0, lMax: 50, aMin: -128, aMax: 128, bMin: -128, bMax: 128 }
        },
        {
            name: 'vivid',
            bounds: { lMin: 25, lMax: 75, aMin: -128, aMax: 128, bMin: -128, bMax: 128 },
            condition: (color) => Math.sqrt(color.a * color.a + color.b * color.b) > 60
        },
        {
            name: 'neutral',
            bounds: { lMin: 0, lMax: 100, aMin: -30, aMax: 30, bMin: -30, bMax: 30 }
        },
        {
            name: 'warm',
            bounds: { lMin: 0, lMax: 100, aMin: 0, aMax: 128, bMin: 0, bMax: 128 }
        },
        {
            name: 'cool',
            bounds: { lMin: 0, lMax: 100, aMin: -128, aMax: 0, bMin: -128, bMax: 0 }
        }
    ];
}

isColorInRegion(color, region) {
    const { bounds, condition } = region;
    const inBounds = color.l >= bounds.lMin && color.l <= bounds.lMax &&
                    color.a >= bounds.aMin && color.a <= bounds.aMax &&
                    color.b >= bounds.bMin && color.b <= bounds.bMax;

    return inBounds && (!condition || condition(color));
}

calculateTotalGamutCoverage(labColors) {
    const convexHull = this.calculateConvexHull(labColors);
    const volume = this.calculateConvexHullVolume(convexHull);
    const maxVolume = 100 * 256 * 256; // Approximate LAB color space volume
    return volume / maxVolume;
}

analyzeGamutCoverageStats(coverage) {
    const values = Array.from(coverage.values());
    return {
        mean: mathUtils.calculateMean(values),
        std: Math.sqrt(mathUtils.calculateVariance(values)),
        uniformity: this.calculateCoverageUniformity(values)
    };
}

calculateCoverageUniformity(values) {
    const mean = mathUtils.calculateMean(values);
    const maxDeviation = values.reduce((max, val) => 
        Math.max(max, Math.abs(val - mean)), 0
    );
    return 1 - (maxDeviation / mean);
}

calculatePerceptualSpread(labColors) {
    const distances = [];
    for (let i = 0; i < labColors.length - 1; i++) {
        for (let j = i + 1; j < labColors.length; j++) {
            distances.push(mathUtils.deltaE(labColors[i], labColors[j]));
        }
    }

    return {
        mean: mathUtils.calculateMean(distances),
        median: this.calculateMedian(distances),
        std: Math.sqrt(mathUtils.calculateVariance(distances)),
        max: Math.max(...distances),
        distribution: this.calculateDistanceDistribution(distances)
    };
}

calculateDistanceDistribution(distances) {
    const binCount = 20;
    const max = Math.max(...distances);
    const binSize = max / binCount;
    const histogram = new Array(binCount).fill(0);

    distances.forEach(distance => {
        const bin = Math.min(binCount - 1, Math.floor(distance / binSize));
        histogram[bin]++;
    });

    const total = distances.length;
    return histogram.map(count => count / total);
}

analyzeColorHarmonyInLab(labColors) {
    const complementaryPairs = this.findComplementaryPairsLab(labColors);
    const analogousSets = this.findAnalogousSetsLab(labColors);
    const triadicSets = this.findTriadicSetsLab(labColors);

    return {
        complementary: complementaryPairs,
        analogous: analogousSets,
        triadic: triadicSets,
        quality: this.assessHarmonyQuality({
            complementary: complementaryPairs,
            analogous: analogousSets,
            triadic: triadicSets
        })
    };
}

assessHarmonyQuality(harmonies) {
    const complementaryScore = this.assessHarmonySet(harmonies.complementary);
    const analogousScore = this.assessHarmonySet(harmonies.analogous);
    const triadicScore = this.assessHarmonySet(harmonies.triadic);

    return complementaryScore * 0.4 + analogousScore * 0.3 + triadicScore * 0.3;
}

rgbToHSV(r, g, b) {
    r /= 255;
    g /= 255;
    b /= 255;

    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const diff = max - min;

    let h = 0;
    const s = max === 0 ? 0 : diff / max;
    const v = max;

    if (diff !== 0) {
        switch (max) {
            case r:
                h = ((g - b) / diff) % 6;
                break;
            case g:
                h = (b - r) / diff + 2;
                break;
            default: // b is max
                h = (r - g) / diff + 4;
        }
        h *= 60;
        if (h < 0) h += 360;
    }

    return { h: h / 360, s, v };
}

assessLABQuality(labColors) {
    const distributionQuality = this.assessDistributionQuality(labColors);
    const contrastQuality = this.assessContrastQuality(labColors);
    const harmonyQuality = this.assessHarmonyQualityLab(labColors);

    return {
        overall: distributionQuality * 0.4 + contrastQuality * 0.3 + harmonyQuality * 0.3,
        components: {
            distribution: distributionQuality,
            contrast: contrastQuality,
            harmony: harmonyQuality
        }
    };
}

calculateLABStatistics(labColors) {
    if (!labColors || !labColors.length) {
        return this.getDefaultLABStatistics();
    }

    const channels = ['l', 'a', 'b'];
    const stats = {};

    channels.forEach(channel => {
        const values = labColors.map(color => color[channel]);
        const mean = mathUtils.calculateMean(values);
        const variance = mathUtils.calculateVariance(values);

        stats[channel] = {
            mean,
            std: Math.sqrt(variance),
            min: Math.min(...values),
            max: Math.max(...values),
            median: this.calculateMedian(values)
        };
    });

    return stats;
}

getDefaultLABStatistics() {
    return {
        l: { mean: 50, std: 0, min: 0, max: 100, median: 50 },
        a: { mean: 0, std: 0, min: -128, max: 127, median: 0 },
        b: { mean: 0, std: 0, min: -128, max: 127, median: 0 }
    };
}


    static getDefaultCoherenceAnalysis() {
        return {
            vectors: [],
            distribution: {},
            metrics: {
                overallCoherence: 0,
                componentDistribution: {},
                spatialCoverage: 0
            }
        };
    }

 getDefaultSpatialAnalysis() {
        return {
            regions: [],
            relationships: [],
            density: 0,
            distribution: {},
            quality: {
                overall: 0,
                components: {
                    spatial: 0,
                    relational: 0
                }
            }
        };
    }
}

module.exports = ColorAnalysis;
