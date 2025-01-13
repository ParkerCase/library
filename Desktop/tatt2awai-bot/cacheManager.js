let LRU;
try {
    LRU = require('lru-cache');
} catch (err) {
    // Fallback implementation if LRU-cache is not available
    LRU = class FallbackCache {
        constructor() {
            this.store = new Map();
        }

        get(key) {
            return this.store.get(key);
        }

        set(key, value) {
            this.store.set(key, value);
            return true;
        }

        del(key) {
            return this.store.delete(key);
        }
    };
}

class CacheManager {
    constructor() {
        try {
            // Try to create LRU cache with config if available
            const defaultConfig = {
                max: 500,  // Default to 500 items
                maxAge: 1000 * 60 * 60  // Default to 1 hour
            };

            let cacheConfig = defaultConfig;

            try {
                const config = require('./config');
                if (config.optimization && config.optimization.caching) {
                    cacheConfig = {
                        max: parseInt(config.optimization.caching.maxSize) || defaultConfig.max,
                        maxAge: (config.optimization.caching.ttl * 1000) || defaultConfig.maxAge
                    };
                }
            } catch (configErr) {
                console.warn('Config not available, using default cache settings');
            }

            this.cache = new LRU(cacheConfig);
        } catch (err) {
            console.warn('Failed to initialize LRU cache, using fallback implementation');
            this.cache = new Map();
        }
    }

    async get(key) {
        try {
            return this.cache.get(key);
        } catch (err) {
            console.error('Cache get operation failed:', err);
            return null;
        }
    }

    async set(key, value) {
        try {
            return this.cache.set(key, value);
        } catch (err) {
            console.error('Cache set operation failed:', err);
            return false;
        }
    }

    async invalidate(key) {
        try {
            return this.cache instanceof Map ? 
                this.cache.delete(key) : 
                this.cache.del(key);
        } catch (err) {
            console.error('Cache invalidate operation failed:', err);
            return false;
        }
    }

    async clear() {
        try {
            if (this.cache instanceof Map) {
                this.cache.clear();
            } else {
                this.cache.reset();
            }
            return true;
        } catch (err) {
            console.error('Cache clear operation failed:', err);
            return false;
        }
    }
}

module.exports = new CacheManager();
