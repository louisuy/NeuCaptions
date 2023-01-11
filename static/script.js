document.addEventListener("DOMContentLoaded", function(){
    document.querySelector("#upload-button").addEventListener("click", function() {
        console.log('Hello world!')
        document.querySelector("#fileInput").value = ""; // reset file input
        document.querySelector("#fileInput").click();
    });
    document.querySelector("#fileInput").addEventListener("change", function() {
        var fileName = this.value.split("\\").pop();
        var file    = document.getElementById('fileInput').files[0];
        var reader  = new FileReader();
      
        reader.onloadend = function () {
            document.querySelector("#bg-image").style.background="url("+ reader.result +")"; 
            document.querySelector("#bg-image").classList.add("blurred");
            document.querySelector("#bg-image").style.backgroundSize = "cover";
            document.querySelector(".square").innerHTML = "<img src='" + reader.result + "'/>";
        }
      
        if (file) {
          reader.readAsDataURL(file);
        } else {
        //   
        }
        // sessionStorage.setItem('bg-image', reader.result);
      }
    ,);
});
