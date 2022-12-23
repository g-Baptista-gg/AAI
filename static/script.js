var arr = Array.from({length: 3000}, (item, index) => index);
var y = Array.of(3000)

for (let i = 0; i < 3000; i++) {
    y[i] = 1;
}

function plotData() {
    var data = {
        // A labels array that can contain any sort of values. It will be your x_labels
        labels: arr,
        // Our series array that contains series objects or in this case series data arrays
        series: [
            y
        ]
        };

        var options = {
            width: '800px',
            height: '300px',
        };

    var chart = new Chartist.Line('#chart', data, options);
};

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
        document.getElementById("result").innerHTML = "PLEASE, MAKE A MOVE";

}

function resultGame(player, pc) {
    if (player == pc)
        document.getElementById("result").innerHTML = "DRAW";
    else if (player == (pc - 1) || (player == 3 && pc == 1)) {
        document.getElementById("result").innerHTML = "YOU LOST :(";
        pcScore++;
        document.getElementById("pcScore").innerHTML = pcScore;
    }
    else {
        document.getElementById("result").innerHTML = "EZ GAME :P";
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
        y = msg.signal;
        for (var i = 0; i < length; i++)
            numberArray.push(parseInt(y[i]));
        chart.data.series[0] = y;
        chart.update()
        socket.close()
    });

    // This handles any error, like connection drops.
    socket.on('connect_error', function (event) {
        alert('Can not connect to the server.\nIs the raspberry IP correct?\nIs the server running?\nAre you connect to the "PI" network?');
        console.log('Loss of connection event');
        clearInterval(interval);
        socket.close();
    });
};