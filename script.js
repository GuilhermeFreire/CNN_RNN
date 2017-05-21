(function(){
    var current = 0;
    var frases = "";

    function getData(url, callback) {
        var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {
            if (this.readyState == 4 && this.status == 200) {
                callback(this.responseText);
            }
        };
        xhttp.open("GET", url, true);
        xhttp.send();
    }

    function postData(url, data, callback) {
        var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {
            if (this.readyState == 4 && this.status == 200) {
                callback(this.responseText);
            }
        };
        xhttp.open("POST", url, true);
        xhttp.setRequestHeader("Content-type", "application/json");
        xhttp.send(data);
    }

    function shuffle(array) {
        var currentIndex = array.length, temporaryValue, randomIndex;

        // While there remain elements to shuffle...
        while (0 !== currentIndex) {

            // Pick a remaining element...
            randomIndex = Math.floor(Math.random() * currentIndex);
            currentIndex -= 1;

            // And swap it with the current element.
            temporaryValue = array[currentIndex];
            array[currentIndex] = array[randomIndex];
            array[randomIndex] = temporaryValue;
        }

        return array;
    }

    function nextImage(){
        if(current >= frases.length){
            document.querySelector(".send").style.display = "none";
            //TODO: Notificar fim das imagens

            document.querySelector("#container").style.display = "none";
            document.querySelector("#message").classList.remove("hidden");
            return
        }
        while(frases[current][0] == "" || frases[current][1] == "" || frases[current][2] == "" || frases[current][3] == ""){
            if(current >= frases.length){
                document.querySelector(".send").style.display = "none";
                //TODO: Notificar fim das imagens

                document.querySelector("#container").style.display = "none";
                document.querySelector("#message").classList.remove("hidden");
                return
            }
            current++;
            console.log("jump");
        }

        document.querySelector("#input_image").src = "./imgs/"+(frases[current][0]);

        var setenceIds = [
            "#sentence_1",
            "#sentence_2",
            "#sentence_3"
        ] ;

        shuffle(setenceIds);

        document.querySelector(setenceIds[0]).innerHTML = frases[current][1];
        document.querySelector(setenceIds[0]).dataset.sequence = 1;
        document.querySelector(setenceIds[1]).innerHTML = frases[current][2];
        document.querySelector(setenceIds[1]).dataset.sequence = 2;
        document.querySelector(setenceIds[2]).innerHTML = frases[current][3];
        document.querySelector(setenceIds[2]).dataset.sequence = 3;

        current++;
    }

    getData("./frases.json", function (json) {
        frases = JSON.parse(json); //TODO: try catch
        shuffle(frases);

        nextImage();
    })

    for(var i = 1; i<=3; i++){
        document.querySelector("#sentence_"+i).addEventListener("dragstart", function(ev){
            ev.dataTransfer.setData("array", JSON.stringify([ev.target.id, ev.target.dataset.sequence, ev.target.innerHTML]));
        });

        document.querySelector("#pos"+i).addEventListener("dragover", function(ev){
            ev.preventDefault();
        });

        document.querySelector("#pos"+i).addEventListener("drop", function(ev){
            ev.preventDefault();
            var data = JSON.parse(ev.dataTransfer.getData("array"));
            var sentenceElement = document.querySelector("#"+data[0]);
            sentenceElement.style.opacity = 0.2;
            sentenceElement.setAttribute('draggable', false);
            var target = ev.target.parentNode.parentNode;

            if(target.querySelector(".content").innerHTML != ""){
                var oldSentence = document.querySelector("#"+target.dataset.pId);
                oldSentence.style.opacity = 1;
                oldSentence.setAttribute('draggable', true);
            }
            target.dataset.sequence = data[1];
            target.dataset.pId = data[0];
            target.querySelector(".content").innerHTML = data[2];
        });
    }

    document.querySelector("#delete_1").addEventListener("click", function(){
        var pos = document.querySelector("#pos1");
        var oldSentence = document.querySelector("#"+pos.dataset.pId);
        oldSentence.style.opacity = 1;
        oldSentence.setAttribute('draggable', true);
        delete pos.dataset.sequence;
        delete pos.dataset.pId;
        pos.querySelector(".content").innerHTML = "";
    });

    document.querySelector("#delete_2").addEventListener("click", function(){
        var pos = document.querySelector("#pos2");
        var oldSentence = document.querySelector("#"+pos.dataset.pId);
        oldSentence.style.opacity = 1;
        oldSentence.setAttribute('draggable', true);
        delete pos.dataset.sequence;
        delete pos.dataset.pId;
        pos.querySelector(".content").innerHTML = "";
    });

    document.querySelector("#delete_3").addEventListener("click", function(){
        var pos = document.querySelector("#pos3");
        var oldSentence = document.querySelector("#"+pos.dataset.pId);
        oldSentence.style.opacity = 1;
        oldSentence.setAttribute('draggable', true);
        delete pos.dataset.sequence;
        delete pos.dataset.pId;
        pos.querySelector(".content").innerHTML = "";
    });

    document.querySelector(".send").addEventListener("click", function(ev){
        ev.disabled = true;
        var data = {
            image: document.querySelector("#input_image").src,
            first: document.querySelector("#pos1").dataset.sequence,
            second: document.querySelector("#pos2").dataset.sequence,
            third: document.querySelector("#pos3").dataset.sequence,
        };

        if (data.image == "" || data.image == undefined || data.first == undefined || data.second == undefined || data.third == undefined){
            //TODO: Alert
            ev.disabled = false;
            return
        }

        postData("/sendResults", JSON.stringify(data), function (response) {
            response = JSON.parse(response);
            if(response.success == true){
                document.querySelector("#delete_1").click();
                document.querySelector("#delete_2").click();
                document.querySelector("#delete_3").click();
                nextImage()
            }else{
                //TODO: Alert
            }
            ev.disabled = false;
        })
    })
})();