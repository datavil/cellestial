document.addEventListener("DOMContentLoaded", function () {
    console.log("Cellestial documentation: soft styling and parameter formatting active.");

    // Replace the annoying en-dash (\u2013) with a newline and indentation in parameter lists
    const paramDescriptions = document.querySelectorAll('.field-list dd li p, .field-list dd p');
    paramDescriptions.forEach(p => {
        // Use regex for en-dash (\u2013) with surrounding spaces
        const enDashRegex = / \u2013 /g;
        if (p.innerHTML.match(enDashRegex)) {
            p.innerHTML = p.innerHTML.replace(enDashRegex, '<br>&nbsp;&nbsp;&nbsp;&nbsp;');
        }
    });

    // Ensure signatures have appropriate padding and layout
    const sigs = document.querySelectorAll('dt.sig');
    sigs.forEach(sig => {
        // Any dynamic adjustments can go here
    });
});
