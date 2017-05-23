$(function() {
    var drawboardCanvas = document.getElementById('drawboardCanvas');
    preventPageScrolling(drawboardCanvas);

    var curColor = $('#selectColor option:selected').val();
    var currentChar = null;
    var blockUserInput = false;

    if(drawboardCanvas) {
        var isDown = false;
        var ctx = drawboardCanvas.getContext('2d');
        var canvasX, canvasY;
        //'pencil' details
        ctx.lineWidth = 40.0;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.strokeStyle = curColor;
        var lastX = lastY = 0

        $(drawboardCanvas)
            .on('touchstart mousedown', function(e) {
                if (blockUserInput)
                    return;

                isDown = true;
                ctx.beginPath();
                var pos = getTouchPos(e)
                canvasX = pos.x - drawboardCanvas.offsetLeft;
                canvasY = pos.y - drawboardCanvas.offsetTop;
                ctx.moveTo(canvasX, canvasY);
                ctx.lineTo(canvasX, canvasY+1);  // +1 required for iOS-Safari
                lastX = canvasX;
                lastY = canvasY;
                ctx.stroke();
            });
        $(drawboardCanvas)
            .on('touchmove mousemove', function(e) {
                if (blockUserInput)
                    return;

                if(isDown != false) {
                    ctx.beginPath();
                    var pos = getTouchPos(e)
                    canvasX = pos.x - drawboardCanvas.offsetLeft;
                    canvasY = pos.y - drawboardCanvas.offsetTop;
                    ctx.moveTo(lastX, lastY)
                    ctx.lineTo(canvasX, canvasY+1); // +1 required for iOS-Safari
                    lastX = canvasX;
                    lastY = canvasY;
                    ctx.stroke();
                }
            });
        $(drawboardCanvas)
            .on('touchend mouseup touchleave mouseleave', function(e) {
                if (blockUserInput)
                    return;

                if (isDown) {
                    setButtonState(true);
                    isDown = false;
                }

                isDown = false;
            });
        $(document)
            .on('keyup', function(e){
                if (blockUserInput)
                    return;

                if (e.keyCode == 27) { // ESC
                    clearCanvas(drawboardCanvas);
                }
            });
        
        $('#submit').click(function(e) {
            setButtonState(false);

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

            setSubmitButtonAnimationState(true);
            blockUserInput = true;

            $.ajax({
                type: 'POST',
                url: 'http://localhost:64303/api/handwriting',
                //url: 'http://bsautermeister.de/handwriting-service/api/handwriting',
                crossDomain: true,
                contentType: 'application/json',
                dataType: 'json',
                data: JSON.stringify({ 'img': scaledImg, 'label': currentChar }),
                success: function(data, status, xhr) {
                    if (status == 'success') {
                        restart();
                    } else {
                        alert('Post Failed: ' + status);
                        restart();
                    }

                    // sleep time expects milliseconds
                    function sleep (time) {
                      return new Promise((resolve) => setTimeout(resolve, time));
                    }

                    // Usage!
                    sleep(1000).then(() => {
                        // Do something after the sleep!
                        setSubmitButtonAnimationState(false);
                        blockUserInput = false;
                    });


                },
                error: function(data, status, err) {
                    setSubmitButtonAnimationState(false);
                    blockUserInput = false;

                    alert('Post Error: ' + status);
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

function setSubmitButtonAnimationState(enabled) {
    if (enabled) {
        $('#submit-icon').removeClass('glyphicon-ok')
        $('#submit-icon').addClass('glyphicon-refresh')
        $('#submit-icon').addClass('glyphicon-animate')
    } else {
        $('#submit-icon').addClass('glyphicon-ok')
        $('#submit-icon').removeClass('glyphicon-refresh')
        $('#submit-icon').removeClass('glyphicon-animate')
    }
}

function getRandomChar() {
    return String.fromCharCode(Math.floor((Math.random() * 26) + 65));
}

function clearCanvas(canvas) {
    var ctx = canvas.getContext('2d');
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

function preventPageScrolling(canvas) {
    // Prevent scrolling when touching the canvas on iOS-Safari
    document.body.addEventListener('touchstart', function (e) {
      if (e.target == canvas) {
        e.preventDefault();
      }
    }, false);
    document.body.addEventListener('touchend', function (e) {
      if (e.target == canvas) {
        e.preventDefault();
      }
    }, false);
    document.body.addEventListener('touchmove', function (e) {
      if (e.target == canvas) {
        e.preventDefault();
      }
    }, false);
}

function getTouchPos(e) {
    var posX = e.pageX || e.originalEvent.touches[0].pageX;
    var posY = e.pageY || e.originalEvent.touches[0].pageY;
    return Object.freeze({ x: posX, y: posY });
}
