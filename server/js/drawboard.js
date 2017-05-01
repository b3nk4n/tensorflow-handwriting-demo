$(function() {
    var drawboardCanvas = document.getElementById("drawboardCanvas");
    var curColor = $('#selectColor option:selected').val();
    var currentChar = null;

    if(drawboardCanvas) {
        var isDown = false;
        var ctx = drawboardCanvas.getContext("2d");
        var canvasX, canvasY;
        //'pencil' details
        ctx.lineWidth = 20.0;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = curColor;
            
        $(drawboardCanvas)
            .bind( "touchstart mousedown", function(e) {
                isDown = true;
                ctx.beginPath();
                canvasX = e.pageX - drawboardCanvas.offsetLeft;
                canvasY = e.pageY - drawboardCanvas.offsetTop;
                ctx.moveTo(canvasX, canvasY);
                ctx.lineTo(canvasX, canvasY);
                ctx.stroke();
            });
        $(drawboardCanvas)
            .bind( "touchmove mousemove", function(e) {
                if(isDown != false) {
                    canvasX = e.pageX - drawboardCanvas.offsetLeft;
                    canvasY = e.pageY - drawboardCanvas.offsetTop;
                    ctx.lineTo(canvasX, canvasY);
                    ctx.stroke();
                }
            });
        $(drawboardCanvas)
            .bind( "touchend mouseup touchleave mouseleave", function(e) {
                if (isDown) {
                    setButtonState(true);
                    isDown = false;
                }

                isDown = false;
                ctx.closePath();
            });
        $(document)
            .bind( "keyup", function(e){
                if (e.keyCode == 27) { // ESC
                    clearCanvas(drawboardCanvas);
                }
            });
        
        $('#submit').click(function(e) {
            var imgData = ctx.getImageData(0, 0, drawboardCanvas.width, drawboardCanvas.height)

            var SCALE = 10;
            var scaledImg = [];
            for (var y = 0; y < imgData.height; y += SCALE)
            {
                for (var x = 0; x < imgData.width; x += SCALE)
                {
                   scaledImg.push(getPixelGroupValue(imgData, x, y, SCALE, SCALE));
                }
            }

            $.ajax({
                type: "POST",
                url: "http://localhost:3000/api/handwriting",
                crossDomain: true,
                contentType: 'application/json',
                dataType: 'json',
                data: JSON.stringify({ 'img': scaledImg, 'label': currentChar }),
                success: function(data, status, xhr) {
                    if (status == "success") {
                        restart();
                    } else {
                        console.log(status);
                        restart();
                    }
                },
                error: function(data, status, err) {
                    alert('Post Failed: ' + status)
                }
            });
        });

        $('#clear').click(function(e) {
            clearCanvas(drawboardCanvas);
        });

        restart();
    }
        
    $('#selectColor').change(function () {
        curColor = $('#selectColor option:selected').val();
    });

    function restart() {
        var nextChar;
        do {
            nextChar = getRandomChar();
        } while (currentChar == nextChar);
        currentChar = nextChar
        $('#character').text(currentChar);
        clearCanvas(drawboardCanvas);
        setButtonState(false);
    }
});

function setButtonState(enabled) {
    if (enabled) {
        $('#submit').removeAttr('disabled');
        $('#clear').removeAttr('disabled');
    } else {
        $('#submit').attr('disabled','disabled');
        $('#clear').attr('disabled','disabled');
    }
}

function getRandomChar() {
    return String.fromCharCode(Math.floor((Math.random() * 26) + 65));
}

function clearCanvas(canvas) {
    var ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setButtonState(false);
}

function isPixelSet(imgData, x, y) {
    var index = y * (imgData.width * 4) + 4 * x;

    var red = imgData.data[index];
    var green = imgData.data[index + 1];
    var blue = imgData.data[index + 2];
    var alpha = imgData.data[index + 3];
    return alpha > 0;
}

function getPixelGroupValue(imgData, x, y, width, height) {
    var total = width * height;
    var pixCounter = 0.0;
    for (var r = y; r < y + height; r++) {
        for (var c = x; c < x + width; c++) {
            if (isPixelSet(imgData, c, r))
                pixCounter++;
        }
    }
    return pixCounter / total;
}