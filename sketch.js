var inpX = [];
var targetY = [];
var n;
var y1 = 0;
var y2 = 0;
function setup() {
	createCanvas(500, 500);
	y1 = height;
	// frameRate(30);
	n = new NeuralNetwork(1, 2, 1);
}
function draw() {
	background(0);
	strokeWeight(4);
	stroke("white");
	line(0, y1, width, y2);
	// console.log(n.training_complete)
	if (inpX.length != 0) {
		train(inpX, targetY);

		var lineY = guess([[0], [1]]);
		y1 = map(lineY[0], 1, 0, 0, height);
		y2 = map(lineY[1], 1, 0, 0, height);
		strokeWeight(4);

		displayPoints();
	}
}

function guess(inputs) {
	return tf.tidy(() => {
		try {
			inputs = tf.tensor2d(inputs);
			return n.model.predict(inputs).dataSync();
		} catch (e) {
			return null;
		}
	});
}

function train(inputs, targets) {
	// tf.tidy(() => {
	if (n.training_complete) {
		n.train(inputs, targets);
	}
	// });
}

function mouseClicked() {
	console.log("clicked");
	var x = map(mouseX, 0, width, 0, 1);
	var y = map(mouseY, 0, height, 1, 0);

	targetY.push([y]);
	inpX.push([x]);
	// console.log(inpX);
	// console.log(targetY);
}

function displayPoints() {
	stroke("white");
	strokeWeight(8);
	for (var i = 0; i < inpX.length; i++) {
		var x = map(inpX[i], 0, 1, 0, width);
		var y = map(targetY[i], 1, 0, 0, height);
		point(x, y);
	}
}
