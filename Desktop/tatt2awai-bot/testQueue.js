const Queue = require('bull');
const testQueue = new Queue('test-queue', {
  redis: { host: 'localhost', port: 6379 },
});

testQueue.add({ test: 'data' }).then(() => console.log('Job added successfully')).catch(console.error);
testQueue.process((job) => {
  console.log('Processing job:', job.data);
  return Promise.resolve();
});
