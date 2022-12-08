var images = [], x = -1;
images[0] = "static/classification_game_stone.png";
images[1] = "static/classification_game_paper.png";
images[2] = "static/classification_game_scissor.png";

function playRPS(i) {
    document.getElementById("img").src = images[i];
    var pc = Math.random();
    if (pc < 0.33)
        document.getElementById("imgpc").src = images[0];
    else if (pc < 0.66)
        document.getElementById("imgpc").src = images[1];
    else
        document.getElementById("imgpc").src = images[2];
    resultGame()
}

function resultGame() {
    if (document.getElementById("img").src == document.getElementById("imgpc").src)
        document.getElementById("result").innerHTML = "EMPATE";
    else
        document.getElementById("result").innerHTML = "U ARE POOPOO";
}

// To help you connecting to the RPi, the input form value is populated automatically using the current URL.
window_url = window.location.href.split(':')[1].substring(2)
document.getElementById("raspIP").value = window_url

function raspConnect(){
    // Initialize the socket
    var socket = io(document.getElementById("raspIP").value + ':5000/');

    // This function is called after the socket is initiated
    socket.on('connect', function() {
        // Upon a successful connection it starts sending data requests periodically to the server
        interval = setInterval(function(){
            socket.emit('sendData', JSON.stringify({type:'sendData'}))
        }, 1000);
    
        console.log("DEBUG: A sendData request was sent to the server.");
    });

    // This function reads the server message and updates the random integer in the web page
    socket.on('serverResponse', function(msg) {
        console.log("DEBUG: A serverResponse was received by the client.");
        document.getElementById("random_integer").innerHTML = msg.data;
    });

    // This handles any error, like connection drops.
    socket.on('connect_error', function (event) {
        alert('Can not connect to the server.\nIs the raspberry IP correct?\nIs the server running?\nAre you connect to the "PI" network?');
        console.log('Loss of connection event');
        clearInterval(interval);
        socket.close();
    });

}