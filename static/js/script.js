document.addEventListener("DOMContentLoaded", function () {
    let images = JSON.parse(document.getElementById("image-data").textContent);
    let firstImage = JSON.parse(document.getElementById("first-image").textContent);

    if (!firstImage || firstImage === "undefined") {
        firstImage = images.length > 0 ? images[0] : "default.png";
    }

    let index = images.indexOf(firstImage);
    if (index === -1) index = 0;

    let displayImage = document.getElementById('display-image');
    let selectedImageName = document.getElementById('selected-image-name');
    let galleryContainer = document.getElementById("gallery-container");
    let plotContainer = document.querySelector(".right"); 

    // For run_model device/codes
    let deviceLine = document.getElementById("device-line");
    let codesLine = document.getElementById("codes-line");

    // Spinner
    let spinner = document.createElement("div");
    spinner.classList.add("spinner");
    spinner.innerHTML = `<div class="spinner-circle"></div><p>Loading analysis...</p>`;

    // Track if current plot is Visualization or not
    window.isVisualizationPlot = false;
    window.currentBlockId = null;

    window.lastImageWithModelInfo = null;

    // Helper to disable/enable all buttons
    function disableAllButtons() {
        document.querySelectorAll("button").forEach(btn => {
            btn.disabled = true;
        });
    }
    function enableAllButtons() {
        document.querySelectorAll("button").forEach(btn => {
            btn.disabled = false;
        });
    }

    function clearPlotImages() {
        plotContainer.innerHTML = `
          <div class="modern-box">
            <h2 class="modern-subtitle">Generated Plots</h2>
          </div>
          <p style="font-family:'Comic Sans MS'; font-weight:bold; font-size:16px;">
            No plots generated yet.
          </p>
        `;
        console.log("🧹 Cleared Generated Plots");
    }

    function showSpinner() {
        plotContainer.appendChild(spinner);
        disableAllButtons();  // disable everything while loading
    }
    function hideSpinner() {
        if (plotContainer.contains(spinner)) {
            plotContainer.removeChild(spinner);
        }
        enableAllButtons();
    }

    function updatePlotImage(plotPath) {
        plotContainer.innerHTML = `
          <div class="modern-box">
            <h2 class="modern-subtitle">Generated Plots</h2>
          </div>
        `;
        let img = document.createElement("img");
        img.src = `${plotPath}?t=${new Date().getTime()}`;
        img.alt = "Generated Plot";
        img.className = "plot-image";
        img.addEventListener("click", () => {
            openPlotModal(img.src);
        });
        plotContainer.appendChild(img);
        console.log(`📈 Updated Plot: ${plotPath}`);
    }

    function typePlotMessage(text) {
        plotContainer.innerHTML = `
          <div class="modern-box">
            <h2 class="modern-subtitle">Generated Plots</h2>
          </div>
        `;
        let spacer = document.createElement("div");
        spacer.style.height = "20px";
        plotContainer.appendChild(spacer);

        let messageElem = document.createElement("p");
        messageElem.style.fontFamily = "'Comic Sans MS', sans-serif";
        messageElem.style.fontSize = "16px";
        messageElem.style.fontWeight = "bold";
        messageElem.style.color = "#B71C1C"; 
        plotContainer.appendChild(messageElem);

        let i = 0;
        function typeChar() {
            if (i < text.length) {
                messageElem.textContent += text.charAt(i);
                i++;
                setTimeout(typeChar, 30);
            }
        }
        typeChar();
    }

    function typeFeedbackMessage(text) {
        let oldMsg = document.getElementById("feedback-message-typed");
        if (oldMsg) {
            oldMsg.remove();
        }
        let messageElem = document.createElement("p");
        messageElem.id = "feedback-message-typed";
        messageElem.style.fontFamily = "'Comic Sans MS', sans-serif";
        messageElem.style.fontSize = "16px";
        messageElem.style.fontWeight = "bold";
        messageElem.style.marginTop = "10px";
        messageElem.style.color = "#000";

        plotContainer.appendChild(messageElem);

        let i = 0;
        function typeChar() {
            if (i < text.length) {
                messageElem.textContent += text.charAt(i);
                i++;
                setTimeout(typeChar, 30);
            }
        }
        typeChar();
    }

    function typeText(element, text, callback) {
        element.style.display = "block";
        element.textContent = "";
        let i = 0;
        let speed = 30;
        function typeChar() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(typeChar, speed);
            } else {
                if (callback) callback();
            }
        }
        typeChar();
    }

    // showModelInfo => disable buttons while "Device on" and "Activated Concepts" are typed
    function showModelInfo(device, codesStr, skipCheck, completionCallback) {
        if (!completionCallback) completionCallback = () => {};

        if (!skipCheck && window.lastImageWithModelInfo === device + codesStr) {
            console.log("Skipping device/codes re-display for same details.");
            completionCallback();
            return;
        }

        window.lastImageWithModelInfo = device + codesStr;

        disableAllButtons(); // disable during typing

        function doneTyping() {
            enableAllButtons();
            completionCallback();
        }

        if (device) {
            let devLineText = `Device on: ${device}`;
            typeText(deviceLine, devLineText, () => {
                if (codesStr) {
                    let codesMsg = `Activated Concepts for Each Block: ${codesStr}`;
                    setTimeout(() => {
                        typeText(codesLine, codesMsg, doneTyping);
                    }, 500);
                } else {
                    codesLine.style.display = "none";
                    doneTyping();
                }
            });
        } else {
            deviceLine.style.display = "none";
            if (codesStr) {
                let codesMsg = `Activated Concepts for Each Block: ${codesStr}`;
                setTimeout(() => {
                    typeText(codesLine, codesMsg, doneTyping);
                }, 500);
            } else {
                codesLine.style.display = "none";
                doneTyping();
            }
        }
    }

    function hideModelInfo() {
        deviceLine.style.display = "none";
        codesLine.style.display = "none";
        deviceLine.textContent = "";
        codesLine.textContent = "";
    }

    window.toggleFeedbackSection = function() {
        let feedbackSection = document.getElementById("visualization-feedback-section");
        if (!feedbackSection) return;

        if (feedbackSection.style.display === "none" || feedbackSection.style.display === "") {
            feedbackSection.style.display = "block";
        } else {
            feedbackSection.style.display = "none";
            return;
        }

        let msgDiv = document.getElementById("feedback-not-available");
        let optsDiv = document.getElementById("feedback-options");
        if (!window.isVisualizationPlot) {
            msgDiv.style.display = "block";
            optsDiv.style.display = "none";
        } else {
            msgDiv.style.display = "none";
            optsDiv.style.display = "block";
        }
    };

    function handleFormSubmission(event) {
        event.preventDefault();
        let form = event.target;
        let formData = new FormData(form);
        let endpoint = form.getAttribute("data-action");

        window.isVisualizationPlot = false;

        if (endpoint === "run_model") {
            clearPlotImages();
            hideModelInfo();
            // disable just the run_model button
            form.querySelector('button[type="submit"]').disabled = true;
            console.log("Run Model => cleared old plots (no spinner).");
        } else {
            // For other endpoints => show spinner => disableAll
            clearPlotImages();
            showSpinner();
            console.log(`Analysis => cleared old plots, spinner shown. Endpoint: ${endpoint}.`);
        }

        fetch(`/${endpoint}`, {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (endpoint !== "run_model") {
                hideSpinner();
            }

            if (data.success) {
                console.log(`✅ Success: ${data.message}`);
                let alwaysTyped = (endpoint === "run_model");
                showModelInfo(data.device, data.codes_str, alwaysTyped, () => {
                    if (endpoint === "run_model") {
                        form.querySelector('button[type="submit"]').disabled = false;
                    }
                });

                if (data.plot_path) {
                    updatePlotImage(data.plot_path);
                }

                if (endpoint === "visualization") {
                    window.isVisualizationPlot = true;
                    let blockId = formData.get("block_id");
                    window.currentBlockId = blockId;
                }
            } else {
                console.error(`❌ Error: ${data.message}`);
                typePlotMessage(data.message);
                if (endpoint === "run_model") {
                    form.querySelector('button[type="submit"]').disabled = false;
                } else {
                    hideSpinner();
                }
            }
        })
        .catch(error => {
            if (endpoint !== "run_model") {
                hideSpinner();
            }
            console.error("❌ Error:", error);
            typePlotMessage(`Error: ${error}`);
            if (endpoint === "run_model") {
                form.querySelector('button[type="submit"]').disabled = false;
            }
        });
    }

    document.querySelectorAll("form").forEach(form => {
        let actionPath = form.action.split("/").pop();
        form.setAttribute("data-action", actionPath);
        form.addEventListener("submit", handleFormSubmission);
    });

    // Visualization from block buttons
    window.runVisualization = function(blockId) {
        clearPlotImages();
        showSpinner();

        let imagePath = document.getElementById('selected-image-path-visualization');
        if (!imagePath) {
            console.error("No hidden input found for visualization form");
            hideSpinner();
            return;
        }

        let formData = new FormData();
        formData.append("image_path", imagePath.value);
        formData.append("block_id", blockId);

        fetch("/visualization", {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideSpinner();
            if (data.success) {
                console.log("✅ Visualization success");
                if (data.plot_path) {
                    updatePlotImage(data.plot_path);
                }
                showModelInfo(data.device, data.codes_str, false);
                window.isVisualizationPlot = true;
                window.currentBlockId = blockId;

                let feedbackSection = document.getElementById("visualization-feedback-section");
                if (feedbackSection && feedbackSection.style.display === "block") {
                    let msgDiv = document.getElementById("feedback-not-available");
                    let optsDiv = document.getElementById("feedback-options");
                    msgDiv.style.display = "none";
                    optsDiv.style.display = "block";
                }
            } else {
                console.error(`❌ Error: ${data.message}`);
                typePlotMessage(data.message);
                window.isVisualizationPlot = false;
            }
        })
        .catch(err => {
            hideSpinner();
            console.error("❌ Visualization request failed:", err);
            typePlotMessage("Error: " + err);
            window.isVisualizationPlot = false;
        });
    };

    function updateImage(newImage = null) {
        if (newImage) {
            index = images.indexOf(newImage);
            if (index === -1) index = 0;
        }
        let selectedImage = images[index];
        displayImage.src = `/images/${selectedImage}?t=${new Date().getTime()}`;
        selectedImageName.innerText = selectedImage;
        document.querySelectorAll("input[name='image_path']").forEach(input => {
            input.value = selectedImage;
        });
        console.log(`🔄 Updated Image: ${selectedImage}`);
    }

    window.openPlotModal = function(src) {
        let plotModal = document.getElementById("plotModal");
        let plotModalImage = document.getElementById("plotModalImage");
        plotModalImage.src = src;
        plotModal.style.display = "block";
        setTimeout(() => {
            plotModal.classList.add('open');
        }, 10);
    };

    window.closePlotModal = function() {
        let plotModal = document.getElementById("plotModal");
        plotModal.classList.remove('open');
        setTimeout(() => {
            plotModal.style.display = "none";
        }, 300);
    };

    function loadGalleryImages() {
        galleryContainer.innerHTML = "";
        images.forEach(imageName => {
            let galleryItem = document.createElement("div");
            galleryItem.className = "gallery-item";
            galleryItem.innerHTML = `
                <img src="/images/${imageName}" alt="Preview" loading="lazy" onclick="selectImage('${imageName}')">
                <span>${imageName}</span>
            `;
            galleryContainer.appendChild(galleryItem);
        });
        console.log("📸 Gallery images loaded.");
    }

    window.openGallery = function() {
        loadGalleryImages();
        let galleryModal = document.getElementById("galleryModal");
        let overlay = document.getElementById("overlay");
        galleryModal.style.display = "block";
        overlay.style.display = "block";
        setTimeout(() => {
            galleryModal.classList.add('open');
            overlay.classList.add('open');
        }, 10);
    };

    window.closeGallery = function() {
        let galleryModal = document.getElementById("galleryModal");
        let overlay = document.getElementById("overlay");
        galleryModal.classList.remove('open');
        overlay.classList.remove('open');
        setTimeout(() => {
            galleryModal.style.display = "none";
            overlay.style.display = "none";
        }, 300);
    };

    window.selectImage = function(imageName) {
        let idx = images.indexOf(imageName);
        if (idx !== -1) {
            index = idx;
            updateImage();
        }
        closeGallery();
    };

    window.nextImage = function() {
        index = (index + 1) % images.length;
        updateImage();
    };

    window.prevImage = function() {
        index = (index - 1 + images.length) % images.length;
        updateImage();
    };

    window.toggleAccordion = function(sectionId) {
        const allContents = document.querySelectorAll('.accordion-content');
        allContents.forEach(content => {
            if (content.id === sectionId) {
                if (content.style.display === 'none' || content.style.display === '') {
                    content.style.display = 'block';
                } else {
                    content.style.display = 'none';
                }
            } else {
                content.style.display = 'none';
            }
        });

        if (sectionId !== "visualization-section") {
            window.isVisualizationPlot = false;
            window.currentBlockId = null;
            let feedbackSection = document.getElementById("visualization-feedback-section");
            if (feedbackSection) {
                feedbackSection.style.display = "none";
            }
        }
    };

    window.submitFeedback = function(label) {
        if (!window.isVisualizationPlot || window.currentBlockId === null) {
            alert("Feedback is only available for a visualized block. Please visualize a block first.");
            return;
        }
        let formData = new FormData();
        formData.append("block_id", window.currentBlockId);
        formData.append("feedback_label", label);

        fetch("/save_feedback", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                typeFeedbackMessage(data.message);
            } else {
                typePlotMessage("Error: " + data.message);
            }
        })
        .catch(err => {
            typePlotMessage("Error: " + err);
        });
    };

    // Initialize
    updateImage();
});
