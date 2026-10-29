let video;
let bodyPose;
let connections = [];
let poses = [];
let singlePose;

async function setup() {
  createCanvas(640, 480); // a common webcam resolution

  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();
// ml5.bodyPose is use to load the pbody pose model and store it in a variable 
// let bodypose = ml5.bodyPose(?model, ?options, ?callback);

  bodyPose = await ml5.bodyPose(); 
  
// this mtd is ude to start the pose detection process and run it on the real time video
// bodyPose.detectStart(media, getPoses);

  bodyPose.detectStart(video, gotPoses);

  connections = bodyPose.getSkeleton();
}

function draw() {
  background(220);
  image(video, 0, 0);

  for (let i = 0; i < poses.length; i++) {
    let pose = poses[i];
    for (let j = 0; j < connections.length; j++) {
      let pointAIndex = connections[j][0];
      let pointBIndex = connections[j][1];
      let pointA = pose.keypoints[pointAIndex];
      let pointB = pose.keypoints[pointBIndex];

      if (pointA.confidence > 0.1 && pointB.confidence > 0.1) {
        stroke(255, 0, 0);
        strokeWeight(2);
        line(pointA.x, pointA.y, pointB.x, pointB.y);
      }
    }
  }

  for (let i = 0; i < poses.length; i++) {
    let pose = poses[i];
    for (let j = 0; j < pose.keypoints.length; j++) {
      let keypoint = pose.keypoints[j];
      
      if (keypoint.confidence > 0.1) {
        fill(0, 255, 0);
        noStroke();
        circle(keypoint.x, keypoint.y, 10);
      }
    }
  }
}

function gotPoses(results) {
  poses = results;
  
}
