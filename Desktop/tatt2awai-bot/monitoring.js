// monitoring.js
const EventEmitter = require('events');

class PerformanceMonitor extends EventEmitter {
    constructor() {
        super();
        this.metrics = new Map();
        this.startTime = Date.now();
    }

    startOperation(name) {
        const opId = `${name}_${Date.now()}`;
        this.metrics.set(opId, {
            name,
            startTime: process.hrtime(),
            memory: process.memoryUsage()
        });
        return opId;
    }

    endOperation(opId) {
        const metric = this.metrics.get(opId);
        if (!metric) return;

        const duration = process.hrtime(metric.startTime);
        const memoryDiff = this.calculateMemoryDiff(
            metric.memory,
            process.memoryUsage()
        );

        this.emit('operationComplete', {
            name: metric.name,
            duration: duration[0] * 1000 + duration[1] / 1000000,
            memoryDiff
        });

        this.metrics.delete(opId);
    }

    calculateMemoryDiff(start, end) {
        return {
            heapUsed: end.heapUsed - start.heapUsed,
            heapTotal: end.heapTotal - start.heapTotal,
            external: end.external - start.external
        };
    }
}

module.exports = new PerformanceMonitor();
