$(document).ready(function () {
    $("body").prepend("<div id='header'></div>");
    $('#header').load("/static/header.html");

    $(document).ajaxStart(function () {
        $("#divLoading").css("display", "block");
    });
    $(document).ajaxComplete(function () {
        $("#divLoading").css("display", "none");
    });
    
});