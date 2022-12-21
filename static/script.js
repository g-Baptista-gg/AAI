var images = [], x = -1;
images[0] = "static/classification_game_stone.png";
images[1] = "static/classification_game_paper.png";
images[2] = "static/classification_game_scissor.png";
images[3] = "static/smile.png";

var playerScore = 0
var pcScore = 0
var pcPlay = 0

function babyMode() {
    var pc = Math.random();
    if (pc < 0.33) {
        document.getElementById("imgpc").src = images[0];
        return 1;
    }
    else if (pc < 0.66) {
        document.getElementById("imgpc").src = images[1];
        return 2;
    }
    else {
        document.getElementById("imgpc").src = images[2];
        return 3;
    }
}

function godMode(i) {
    if (i == 3) {
        document.getElementById("imgpc").src = images[0];
        return 1;
    }
    document.getElementById("imgpc").src = images[i];
    return ++i;
}

function playRPS(i) {
    if (i > 0) {
        document.getElementById("img").src = images[i - 1];
        if (document.getElementById("godmode").checked == false)
            pcPlay = babyMode();
        else
            pcPlay = godMode(i);
        resultGame(i, pcPlay);
    }
    else if (i == 0)
        document.getElementById("img").src = images[3];

}

function resultGame(player, pc) {
    if (player == pc)
        document.getElementById("result").innerHTML = "EMPATE";
    else if (player == (pc - 1) || (player == 3 && pc == 1)) {
        document.getElementById("result").innerHTML = "PERDESTE :(";
        pcScore++;
        document.getElementById("pcScore").innerHTML = pcScore;
    }
    else {
        document.getElementById("result").innerHTML = "FÁCIL :P";
        playerScore++;
        document.getElementById("playerScore").innerHTML = playerScore;
    }
}

function raspConnect(){
    // Initialize the socket
    var socket = io(document.getElementById("raspIP").value + ':5000/');

    // This function is called after the socket is initiated
    socket.on('connect', function() {
        console.log("DEBUG: A sendData request was sent to the server.");
        socket.emit('acquire', "Connected!")
    });

    // This function reads the server message and updates the random integer in the web page
    socket.on('serverResponse', function(msg) {
        console.log("DEBUG: A serverResponse was received by the client.");
        playRPS(parseInt(msg.data));
        socket.close()
    });

    // This handles any error, like connection drops.
    socket.on('connect_error', function (event) {
        alert('Can not connect to the server.\nIs the raspberry IP correct?\nIs the server running?\nAre you connect to the "PI" network?');
        console.log('Loss of connection event');
        clearInterval(interval);
        socket.close();
    });
}