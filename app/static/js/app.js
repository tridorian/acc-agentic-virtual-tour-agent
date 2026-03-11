/**
 * app.js: JS code for the ADK Gemini Live API Toolkit demo app.
 */

/**
 * User Information Collection
 */

// Country phone codes data with flag emojis
const COUNTRY_CODES = [
  { name: "Afghanistan", code: "+93", flag: "🇦🇫" },
  { name: "Albania", code: "+355", flag: "🇦🇱" },
  { name: "Algeria", code: "+213", flag: "🇩🇿" },
  { name: "Andorra", code: "+376", flag: "🇦🇩" },
  { name: "Angola", code: "+244", flag: "🇦🇴" },
  { name: "Argentina", code: "+54", flag: "🇦🇷" },
  { name: "Armenia", code: "+374", flag: "🇦🇲" },
  { name: "Australia", code: "+61", flag: "🇦🇺" },
  { name: "Austria", code: "+43", flag: "🇦🇹" },
  { name: "Azerbaijan", code: "+994", flag: "🇦🇿" },
  { name: "Bahamas", code: "+1-242", flag: "🇧🇸" },
  { name: "Bahrain", code: "+973", flag: "🇧🇭" },
  { name: "Bangladesh", code: "+880", flag: "🇧🇩" },
  { name: "Barbados", code: "+1-246", flag: "🇧🇧" },
  { name: "Belarus", code: "+375", flag: "🇧🇾" },
  { name: "Belgium", code: "+32", flag: "🇧🇪" },
  { name: "Belize", code: "+501", flag: "🇧🇿" },
  { name: "Benin", code: "+229", flag: "🇧🇯" },
  { name: "Bhutan", code: "+975", flag: "🇧🇹" },
  { name: "Bolivia", code: "+591", flag: "🇧🇴" },
  { name: "Bosnia and Herzegovina", code: "+387", flag: "🇧🇦" },
  { name: "Botswana", code: "+267", flag: "🇧🇼" },
  { name: "Brazil", code: "+55", flag: "🇧🇷" },
  { name: "Brunei", code: "+673", flag: "🇧🇳" },
  { name: "Bulgaria", code: "+359", flag: "🇧🇬" },
  { name: "Burkina Faso", code: "+226", flag: "🇧🇫" },
  { name: "Burundi", code: "+257", flag: "🇧🇮" },
  { name: "Cambodia", code: "+855", flag: "🇰🇭" },
  { name: "Cameroon", code: "+237", flag: "🇨🇲" },
  { name: "Canada", code: "+1", flag: "🇨🇦" },
  { name: "Cape Verde", code: "+238", flag: "🇨🇻" },
  { name: "Central African Republic", code: "+236", flag: "🇨🇫" },
  { name: "Chad", code: "+235", flag: "🇹🇩" },
  { name: "Chile", code: "+56", flag: "🇨🇱" },
  { name: "China", code: "+86", flag: "🇨🇳" },
  { name: "Colombia", code: "+57", flag: "🇨🇴" },
  { name: "Comoros", code: "+269", flag: "🇰🇲" },
  { name: "Congo", code: "+242", flag: "🇨🇬" },
  { name: "Costa Rica", code: "+506", flag: "🇨🇷" },
  { name: "Croatia", code: "+385", flag: "🇭🇷" },
  { name: "Cuba", code: "+53", flag: "🇨🇺" },
  { name: "Cyprus", code: "+357", flag: "🇨🇾" },
  { name: "Czech Republic", code: "+420", flag: "🇨🇿" },
  { name: "Denmark", code: "+45", flag: "🇩🇰" },
  { name: "Djibouti", code: "+253", flag: "🇩🇯" },
  { name: "Dominica", code: "+1-767", flag: "🇩🇲" },
  { name: "Dominican Republic", code: "+1-809", flag: "🇩🇴" },
  { name: "Ecuador", code: "+593", flag: "🇪🇨" },
  { name: "Egypt", code: "+20", flag: "🇪🇬" },
  { name: "El Salvador", code: "+503", flag: "🇸🇻" },
  { name: "Equatorial Guinea", code: "+240", flag: "🇬🇶" },
  { name: "Eritrea", code: "+291", flag: "🇪🇷" },
  { name: "Estonia", code: "+372", flag: "🇪🇪" },
  { name: "Eswatini", code: "+268", flag: "🇸🇿" },
  { name: "Ethiopia", code: "+251", flag: "🇪🇹" },
  { name: "Fiji", code: "+679", flag: "🇫🇯" },
  { name: "Finland", code: "+358", flag: "🇫🇮" },
  { name: "France", code: "+33", flag: "🇫🇷" },
  { name: "Gabon", code: "+241", flag: "🇬🇦" },
  { name: "Gambia", code: "+220", flag: "🇬🇲" },
  { name: "Georgia", code: "+995", flag: "🇬🇪" },
  { name: "Germany", code: "+49", flag: "🇩🇪" },
  { name: "Ghana", code: "+233", flag: "🇬🇭" },
  { name: "Gibraltar", code: "+350", flag: "🇬🇮" },
  { name: "Greece", code: "+30", flag: "🇬🇷" },
  { name: "Grenada", code: "+1-473", flag: "🇬🇩" },
  { name: "Guatemala", code: "+502", flag: "🇬🇹" },
  { name: "Guinea", code: "+224", flag: "🇬🇳" },
  { name: "Guinea-Bissau", code: "+245", flag: "🇬🇼" },
  { name: "Guyana", code: "+592", flag: "🇬🇾" },
  { name: "Haiti", code: "+509", flag: "🇭🇹" },
  { name: "Honduras", code: "+504", flag: "🇭🇳" },
  { name: "Hong Kong", code: "+852", flag: "🇭🇰" },
  { name: "Hungary", code: "+36", flag: "🇭🇺" },
  { name: "Iceland", code: "+354", flag: "🇮🇸" },
  { name: "India", code: "+91", flag: "🇮🇳" },
  { name: "Indonesia", code: "+62", flag: "🇮🇩" },
  { name: "Iran", code: "+98", flag: "🇮🇷" },
  { name: "Iraq", code: "+964", flag: "🇮🇶" },
  { name: "Ireland", code: "+353", flag: "🇮🇪" },
  { name: "Isle of Man", code: "+44", flag: "🇮🇲" },
  { name: "Israel", code: "+972", flag: "🇮🇱" },
  { name: "Italy", code: "+39", flag: "🇮🇹" },
  { name: "Jamaica", code: "+1-876", flag: "🇯🇲" },
  { name: "Japan", code: "+81", flag: "🇯🇵" },
  { name: "Jordan", code: "+962", flag: "🇯🇴" },
  { name: "Kazakhstan", code: "+7", flag: "🇰🇿" },
  { name: "Kenya", code: "+254", flag: "🇰🇪" },
  { name: "Kiribati", code: "+686", flag: "🇰🇮" },
  { name: "Korea, North", code: "+850", flag: "🇰🇵" },
  { name: "Korea, South", code: "+82", flag: "🇰🇷" },
  { name: "Kosovo", code: "+383", flag: "🇽🇰" },
  { name: "Kuwait", code: "+965", flag: "🇰🇼" },
  { name: "Kyrgyzstan", code: "+996", flag: "🇰🇬" },
  { name: "Laos", code: "+856", flag: "🇱🇦" },
  { name: "Latvia", code: "+371", flag: "🇱🇻" },
  { name: "Lebanon", code: "+961", flag: "🇱🇧" },
  { name: "Lesotho", code: "+266", flag: "🇱🇸" },
  { name: "Liberia", code: "+231", flag: "🇱🇷" },
  { name: "Libya", code: "+218", flag: "🇱🇾" },
  { name: "Liechtenstein", code: "+423", flag: "🇱🇮" },
  { name: "Lithuania", code: "+370", flag: "🇱🇹" },
  { name: "Luxembourg", code: "+352", flag: "🇱🇺" },
  { name: "Macao", code: "+853", flag: "🇲🇴" },
  { name: "Madagascar", code: "+261", flag: "🇲🇬" },
  { name: "Malawi", code: "+265", flag: "🇲🇼" },
  { name: "Malaysia", code: "+60", flag: "🇲🇾" },
  { name: "Maldives", code: "+960", flag: "🇲🇻" },
  { name: "Mali", code: "+223", flag: "🇲🇱" },
  { name: "Malta", code: "+356", flag: "🇲🇹" },
  { name: "Marshall Islands", code: "+692", flag: "🇲🇭" },
  { name: "Mauritania", code: "+222", flag: "🇲🇷" },
  { name: "Mauritius", code: "+230", flag: "🇲🇺" },
  { name: "Mexico", code: "+52", flag: "🇲🇽" },
  { name: "Micronesia", code: "+691", flag: "🇫🇲" },
  { name: "Moldova", code: "+373", flag: "🇲🇩" },
  { name: "Monaco", code: "+377", flag: "🇲🇨" },
  { name: "Mongolia", code: "+976", flag: "🇲🇳" },
  { name: "Montenegro", code: "+382", flag: "🇲🇪" },
  { name: "Morocco", code: "+212", flag: "🇲🇦" },
  { name: "Mozambique", code: "+258", flag: "🇲🇿" },
  { name: "Myanmar", code: "+95", flag: "🇲🇲" },
  { name: "Namibia", code: "+264", flag: "🇳🇦" },
  { name: "Nauru", code: "+674", flag: "🇳🇷" },
  { name: "Nepal", code: "+977", flag: "🇳🇵" },
  { name: "Netherlands", code: "+31", flag: "🇳🇱" },
  { name: "New Zealand", code: "+64", flag: "🇳🇿" },
  { name: "Nicaragua", code: "+505", flag: "🇳🇮" },
  { name: "Niger", code: "+227", flag: "🇳🇪" },
  { name: "Nigeria", code: "+234", flag: "🇳🇬" },
  { name: "North Macedonia", code: "+389", flag: "🇲🇰" },
  { name: "Norway", code: "+47", flag: "🇳🇴" },
  { name: "Oman", code: "+968", flag: "🇴🇲" },
  { name: "Pakistan", code: "+92", flag: "🇵🇰" },
  { name: "Palau", code: "+680", flag: "🇵🇼" },
  { name: "Palestine", code: "+970", flag: "🇵🇸" },
  { name: "Panama", code: "+507", flag: "🇵🇦" },
  { name: "Papua New Guinea", code: "+675", flag: "🇵🇬" },
  { name: "Paraguay", code: "+595", flag: "🇵🇾" },
  { name: "Peru", code: "+51", flag: "🇵🇪" },
  { name: "Philippines", code: "+63", flag: "🇵🇭" },
  { name: "Poland", code: "+48", flag: "🇵🇱" },
  { name: "Portugal", code: "+351", flag: "🇵🇹" },
  { name: "Qatar", code: "+974", flag: "🇶🇦" },
  { name: "Romania", code: "+40", flag: "🇷🇴" },
  { name: "Russia", code: "+7", flag: "🇷🇺" },
  { name: "Rwanda", code: "+250", flag: "🇷🇼" },
  { name: "Saint Kitts and Nevis", code: "+1-869", flag: "🇰🇳" },
  { name: "Saint Lucia", code: "+1-758", flag: "🇱🇨" },
  { name: "Saint Vincent and the Grenadines", code: "+1-784", flag: "🇻🇨" },
  { name: "Samoa", code: "+685", flag: "🇼🇸" },
  { name: "San Marino", code: "+378", flag: "🇸🇲" },
  { name: "Sao Tome and Principe", code: "+239", flag: "🇸🇹" },
  { name: "Saudi Arabia", code: "+966", flag: "🇸🇦" },
  { name: "Senegal", code: "+221", flag: "🇸🇳" },
  { name: "Serbia", code: "+381", flag: "🇷🇸" },
  { name: "Seychelles", code: "+248", flag: "🇸🇨" },
  { name: "Sierra Leone", code: "+232", flag: "🇸🇱" },
  { name: "Singapore", code: "+65", flag: "🇸🇬" },
  { name: "Slovakia", code: "+421", flag: "🇸🇰" },
  { name: "Slovenia", code: "+386", flag: "🇸🇮" },
  { name: "Solomon Islands", code: "+677", flag: "🇸🇧" },
  { name: "Somalia", code: "+252", flag: "🇸🇴" },
  { name: "South Africa", code: "+27", flag: "🇿🇦" },
  { name: "South Sudan", code: "+211", flag: "🇸🇸" },
  { name: "Spain", code: "+34", flag: "🇪🇸" },
  { name: "Sri Lanka", code: "+94", flag: "🇱🇰" },
  { name: "Sudan", code: "+249", flag: "🇸🇩" },
  { name: "Suriname", code: "+597", flag: "🇸🇷" },
  { name: "Sweden", code: "+46", flag: "🇸🇪" },
  { name: "Switzerland", code: "+41", flag: "🇨🇭" },
  { name: "Syria", code: "+963", flag: "🇸🇾" },
  { name: "Taiwan", code: "+886", flag: "🇹🇼" },
  { name: "Tajikistan", code: "+992", flag: "🇹🇯" },
  { name: "Tanzania", code: "+255", flag: "🇹🇿" },
  { name: "Thailand", code: "+66", flag: "🇹🇭" },
  { name: "Timor-Leste", code: "+670", flag: "🇹🇱" },
  { name: "Togo", code: "+228", flag: "🇹🇬" },
  { name: "Tonga", code: "+676", flag: "🇹🇴" },
  { name: "Trinidad and Tobago", code: "+1-868", flag: "🇹🇹" },
  { name: "Tunisia", code: "+216", flag: "🇹🇳" },
  { name: "Turkey", code: "+90", flag: "🇹🇷" },
  { name: "Turkmenistan", code: "+993", flag: "🇹🇲" },
  { name: "Tuvalu", code: "+688", flag: "🇹🇻" },
  { name: "Uganda", code: "+256", flag: "🇺🇬" },
  { name: "Ukraine", code: "+380", flag: "🇺🇦" },
  { name: "United Arab Emirates", code: "+971", flag: "🇦🇪" },
  { name: "United Kingdom", code: "+44", flag: "🇬🇧" },
  { name: "United States", code: "+1", flag: "🇺🇸" },
  { name: "Uruguay", code: "+598", flag: "🇺🇾" },
  { name: "Uzbekistan", code: "+998", flag: "🇺🇿" },
  { name: "Vanuatu", code: "+678", flag: "🇻🇺" },
  { name: "Vatican City", code: "+379", flag: "🇻🇦" },
  { name: "Venezuela", code: "+58", flag: "🇻🇪" },
  { name: "Vietnam", code: "+84", flag: "🇻🇳" },
  { name: "Yemen", code: "+967", flag: "🇾🇪" },
  { name: "Zambia", code: "+260", flag: "🇿🇲" },
  { name: "Zimbabwe", code: "+263", flag: "🇿🇼" }
];

// Store user information
let userInfo = null;
let conversationStartedAt = null;
let sessionEndedAt = null;

// Get user info modal and form elements
const userInfoModal = document.getElementById("userInfoModal");
const userInfoForm = document.getElementById("userInfoForm");
const countryCodeInput = document.getElementById("countryCodeInput");
const countryCodeDropdown = document.getElementById("countryCodeDropdown");
const countryCodeHidden = document.getElementById("countryCode");
const endConversationModal = document.getElementById("endConversationModal");
const closeEndConversationModal = document.getElementById("closeEndConversationModal");
const closeSummaryButton = document.getElementById("closeSummaryButton");
const summarySessionId = document.getElementById("summarySessionId");
const summaryUserName = document.getElementById("summaryUserName");
const summaryUserEmail = document.getElementById("summaryUserEmail");
const summaryDuration = document.getElementById("summaryDuration");

// Initialize custom country code dropdown
if (countryCodeInput && countryCodeDropdown) {
  // Create dropdown options
  COUNTRY_CODES.forEach(country => {
    const option = document.createElement("div");
    option.className = "custom-select-option";
    option.textContent = `${country.flag} ${country.code}`;
    option.dataset.code = country.code;
    option.dataset.country = country.name;
    
    option.addEventListener("click", function() {
      countryCodeInput.value = `${country.flag} ${country.code}`;
      countryCodeHidden.value = country.code;
      countryCodeDropdown.classList.remove("show");
      // Trigger form validation
      countryCodeHidden.dispatchEvent(new Event('change'));
    });
    
    countryCodeDropdown.appendChild(option);
  });

  // Default country code to Singapore.
  const defaultCountry = COUNTRY_CODES.find(country => country.name === "Singapore");
  if (defaultCountry) {
    countryCodeInput.value = `${defaultCountry.flag} ${defaultCountry.code}`;
    countryCodeHidden.value = defaultCountry.code;
  }
  
  // Open dropdown on input focus
  countryCodeInput.addEventListener("focus", function() {
    countryCodeDropdown.classList.add("show");
  });
  
  // Filter dropdown based on input
  countryCodeInput.addEventListener("input", function(e) {
    const searchTerm = e.target.value.toLowerCase();
    const options = countryCodeDropdown.querySelectorAll(".custom-select-option");
    let hasVisibleOption = false;
    
    options.forEach(option => {
      const countryName = option.dataset.country.toLowerCase();
      const code = option.dataset.code.toLowerCase();
      
      if (countryName.includes(searchTerm) || code.includes(searchTerm)) {
        option.style.display = "block";
        hasVisibleOption = true;
      } else {
        option.style.display = "none";
      }
    });
    
    // Show dropdown if there are visible options
    if (hasVisibleOption && searchTerm.length > 0) {
      countryCodeDropdown.classList.add("show");
    } else if (searchTerm.length === 0) {
      // Show all options when input is empty
      options.forEach(option => {
        option.style.display = "block";
      });
      countryCodeDropdown.classList.add("show");
    } else {
      countryCodeDropdown.classList.remove("show");
    }
  });
  
  // Close dropdown when clicking outside
  document.addEventListener("click", function(e) {
    if (!e.target.closest(".custom-select-wrapper")) {
      countryCodeDropdown.classList.remove("show");
    }
  });
  
  // Add keyboard navigation
  countryCodeInput.addEventListener("keydown", function(e) {
    const options = Array.from(countryCodeDropdown.querySelectorAll(".custom-select-option:not([style*='display: none'])"));
    const selectedOption = countryCodeDropdown.querySelector(".custom-select-option.selected");
    let currentIndex = options.indexOf(selectedOption);
    
    switch(e.key) {
      case "ArrowDown":
        e.preventDefault();
        countryCodeDropdown.classList.add("show");
        currentIndex = Math.min(currentIndex + 1, options.length - 1);
        if (options[currentIndex]) {
          options.forEach(opt => opt.classList.remove("selected"));
          options[currentIndex].classList.add("selected");
          options[currentIndex].scrollIntoView({ block: "nearest" });
        }
        break;
      case "ArrowUp":
        e.preventDefault();
        currentIndex = Math.max(currentIndex - 1, 0);
        if (options[currentIndex]) {
          options.forEach(opt => opt.classList.remove("selected"));
          options[currentIndex].classList.add("selected");
          options[currentIndex].scrollIntoView({ block: "nearest" });
        }
        break;
      case "Enter":
        e.preventDefault();
        if (options[currentIndex]) {
          options[currentIndex].click();
        }
        break;
      case "Escape":
        e.preventDefault();
        countryCodeDropdown.classList.remove("show");
        break;
    }
  });
}

// Handle user info form submission
if (userInfoForm) {
  userInfoForm.addEventListener("submit", function(e) {
    e.preventDefault();
    
    // Collect user information
    userInfo = {
      fullName: document.getElementById("fullName").value.trim(),
      identificationNo: document.getElementById("identificationNo").value.trim(),
      countryCode: document.getElementById("countryCode").value.trim(),
      phoneNumber: document.getElementById("phoneNumber").value.trim(),
      emailAddress: document.getElementById("emailAddress").value.trim()
    };

    conversationStartedAt = null;
    sessionEndedAt = null;
    isEndingConversation = false;
    hasEndedConversation = false;
    conversationLog = [];
    
    // Hide the modal
    userInfoModal.classList.add("hidden");

    // Initialize speaker output from this user gesture so agent audio can autoplay.
    ensureAudioOutputStarted().catch((error) => {
      console.warn("Failed to initialize audio output on submit:", error);
    });

    // Connect WebSocket and start the agent
    connectWebsocket();

    // Show chat input UI (text + mic + end conversation)
    const startSessionButton = document.getElementById("startSessionButton");
    if (startSessionButton) startSessionButton.classList.add("hidden");
    const chatInputRow = document.getElementById("chatInputRow");
    if (chatInputRow) chatInputRow.classList.remove("hidden");
    if (startAudioButton) {
      startAudioButton.disabled = false;
    }
    setInputMode("text", true);
    if (endConversationButton) {
      endConversationButton.classList.remove("hidden");
      endConversationButton.disabled = false;
    }
  });
}

/**
 * WebSocket handling
 */

// Connect the server with a WebSocket connection
const userId = "demo-user";
const sessionId = "demo-session-" + Math.random().toString(36).substring(7);
let websocket = null;
let is_audio = false;
let hasRequestedAgentWelcome = false;
let isEndingConversation = false;
let hasEndedConversation = false;

async function requestAgentWelcome() {
  if (hasRequestedAgentWelcome || !userInfo || !websocket || websocket.readyState !== WebSocket.OPEN) {
    return;
  }

  // Ensure audio output is initialized so the first spoken turn is audible.
  await ensureAudioOutputStarted().catch((error) => {
    console.warn("Audio output not ready for welcome message:", error);
  });

  const welcomeMessage = `Hi ${userInfo.fullName}, I am STELLA, your personal assistant at Star Learners! 👋  I'm here to walk you through our centre facilities and support you with any information you need. Are you ready to begin?`;
  // const welcomeMessage = `Hi ${userInfo.fullName}, I am STELLA`;
  const bootstrapPrompt = `Say this greeting exactly in one response, without adding extra text: "${welcomeMessage}"`;
  sendMessage(bootstrapPrompt, { isInternal: true });
  hasRequestedAgentWelcome = true;
  awaitingWelcomeConfirmation = true;
  hasTriggeredVirtualTourIntro = false;
  pendingWelcomeScriptAfterTurn = false;
  pendingVirtualTourVideo = false;
}

// Get checkbox elements for RunConfig options
const enableProactivityCheckbox = document.getElementById("enableProactivity");
const enableAffectiveDialogCheckbox = document.getElementById("enableAffectiveDialog");

// Reconnect WebSocket when RunConfig options change
function handleRunConfigChange() {
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    const runConfigSettings = {
      proactivity: !!enableProactivityCheckbox?.checked,
      affective_dialog: !!enableAffectiveDialogCheckbox?.checked
    };

    addSystemMessage("Reconnecting with updated settings...");
    addConsoleEntry('outgoing', 'Reconnecting due to settings change', runConfigSettings, '🔄', 'system');
    websocket.close();
    // connectWebsocket() will be called by onclose handler after delay
  }
}

// Add change listeners to RunConfig checkboxes
if (enableProactivityCheckbox) {
  enableProactivityCheckbox.addEventListener("change", handleRunConfigChange);
}
if (enableAffectiveDialogCheckbox) {
  enableAffectiveDialogCheckbox.addEventListener("change", handleRunConfigChange);
}

// Build WebSocket URL with RunConfig options as query parameters
function getWebSocketUrl() {
  // Use wss:// for HTTPS pages, ws:// for HTTP (localhost development)
  const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const baseUrl = wsProtocol + "//" + window.location.host + "/ws/" + userId + "/" + sessionId;
  const params = new URLSearchParams();

  // Add proactivity option if checked
  if (enableProactivityCheckbox && enableProactivityCheckbox.checked) {
    params.append("proactivity", "true");
  }

  // Add affective dialog option if checked
  if (enableAffectiveDialogCheckbox && enableAffectiveDialogCheckbox.checked) {
    params.append("affective_dialog", "true");
  }

  const queryString = params.toString();
  return queryString ? baseUrl + "?" + queryString : baseUrl;
}

// Get DOM elements
const messagesDiv = document.getElementById("messages");
const statusIndicator = document.getElementById("statusIndicator");
const statusText = document.getElementById("statusText");
const consoleContent = document.getElementById("consoleContent");
const clearConsoleBtn = document.getElementById("clearConsole");
const openConsoleButton = document.getElementById("openConsoleButton");
const closeConsoleButton = document.getElementById("closeConsoleButton");
const consolePopup = document.getElementById("consolePopup");
const showAudioEventsCheckbox = document.getElementById("showAudioEvents");
const endConversationButton = document.getElementById("endConversationButton");
let currentMessageId = null;
let currentBubbleElement = null;
let currentInputTranscriptionId = null;
let currentInputTranscriptionElement = null;
let currentOutputTranscriptionId = null;
let currentOutputTranscriptionElement = null;
let inputTranscriptionFinished = false; // Track if input transcription is complete for this turn
let hasOutputTranscriptionInTurn = false; // Track if output transcription delivered the response
const INTERRUPTION_KEYWORD = "hey stella";
const INTERRUPTION_UNLOCK_MS = 6000;
const WAKE_REPLY_TEXT = "Yes, I'm here! What can I help you with?";
const ENGLISH_ONLY_NOTICE = "Please use English only.";
const DEFAULT_YOUTUBE_URL = "https://youtu.be/tkhpVEcBfv0";
const QDRANT_VIDEO_SCORE_THRESHOLD = 0.05;
const RESUME_VIDEO_HINT_TEXT = 'Video paused. Say "Please continue to play the video" to resume.';
const REGISTRATION_URL = "https://starlearners.com.sg/admission/register-your-interest/";
const REGISTRATION_PROMPT_TEXT = "Great! You can proceed with registration here to embark on a new journey with us:";
const TEAM_CONNECT_URL = "https://starlearners.com.sg/contact-us/";
const TEAM_WHATSAPP_NUMBER = "6562238808"; // Star Learners WhatsApp number (SG format, no +)
const TOUR_END_QUESTION = "That's the end of the virtual tour! Do you have any questions or anything else you'd like to know?";
// const VIRTUAL_TOUR_INTRO_TEXT = "Great to hear that! I'll take you on a virtual tour of our center. Feel free to explore and learn more about what we offer. If you have any questions along the way, just let me know 😊";
const VIRTUAL_TOUR_INTRO_TEXT = "Great to hear that!😊";
let interruptionUnlockedUntil = 0;
let isAgentSpeaking = false;
const KEYWORD_COOLDOWN_MS = 3000;
let lastKeywordDetectedAt = 0;
const ENGLISH_NOTICE_COOLDOWN_MS = 4000;
let lastEnglishOnlyNoticeAt = 0;
const VIDEO_COMMAND_COOLDOWN_MS = 8000;
let lastVideoCommandAt = 0;
const VIDEO_FALLBACK_SUPPRESS_MS = 15000;
let suppressVideoFallbackUntil = 0;
let suppressAgentOutputForTurn = false;
let suppressAgentOutputResetTimer = null;
let awaitingWelcomeConfirmation = false;
let hasTriggeredVirtualTourIntro = false;
let pendingWelcomeScriptAfterTurn = false;
let pendingVirtualTourVideo = false;
let pendingRegistrationLinkAfterTurn = false;
let hasShownRegistrationPrompt = false;
let floatingVideoContainer = null;
let floatingVideoIframe = null;
let floatingVideoResumeHint = null;
let currentYouTubeWatchUrl = DEFAULT_YOUTUBE_URL;
let hasHandledTourEnd = false;
let postTourFollowupActive = false;
let youtubePlayer = null;
let youtubeApiReadyPromise = null;
let youtubeEndMonitorInterval = null;
let keywordRecognizer = null;
let keywordRecognizerActive = false;
let pendingVirtualTourFallbackTimer = null;
let conversationLog = []; // {role: 'user'|'agent', text: string}

// Helper function to clean spaces between CJK characters
// Removes spaces between Japanese/Chinese/Korean characters while preserving spaces around Latin text
function cleanCJKSpaces(text) {
  // CJK Unicode ranges: Hiragana, Katakana, Kanji, CJK Unified Ideographs, Fullwidth forms
  const cjkPattern = /[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf\uff00-\uffef]/;

  // Remove spaces between two CJK characters
  return text.replace(/(\S)\s+(?=\S)/g, (match, char1) => {
    // Get the character after the space(s)
    const nextCharMatch = text.match(new RegExp(char1 + '\\s+(.)', 'g'));
    if (nextCharMatch && nextCharMatch.length > 0) {
      const char2 = nextCharMatch[0].slice(-1);
      // If both characters are CJK, remove the space
      if (cjkPattern.test(char1) && cjkPattern.test(char2)) {
        return char1;
      }
    }
    return match;
  });
}

function normalizeKeywordText(text) {
  return (text || "")
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function hasInterruptionKeyword(text) {
  return normalizeKeywordText(text).includes(INTERRUPTION_KEYWORD);
}

function normalizeUrlCandidate(url) {
  return (url || "").trim().replace(/[)\],.;!?'"`]+$/g, "");
}

function parseYouTubeTimestamp(rawValue) {
  if (!rawValue) {
    return null;
  }

  const value = String(rawValue).trim().toLowerCase();
  if (!value) {
    return null;
  }

  if (/^\d+$/.test(value)) {
    return parseInt(value, 10);
  }

  const match = value.match(/^(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$/);
  if (!match) {
    return null;
  }

  const hours = parseInt(match[1] || "0", 10);
  const minutes = parseInt(match[2] || "0", 10);
  const seconds = parseInt(match[3] || "0", 10);
  const total = (hours * 3600) + (minutes * 60) + seconds;
  return total > 0 ? total : null;
}

function extractYouTubeVideoId(url) {
  try {
    const parsed = new URL(normalizeUrlCandidate(url));
    const host = parsed.hostname.toLowerCase();
    if (host.includes("youtu.be")) {
      return parsed.pathname.replace("/", "").split("/")[0] || null;
    }
    if (host.includes("youtube.com")) {
      const v = parsed.searchParams.get("v");
      if (v) {
        return v;
      }
      if (parsed.pathname.startsWith("/embed/")) {
        return parsed.pathname.split("/embed/")[1]?.split("/")[0] || null;
      }
    }
    return null;
  } catch (e) {
    return null;
  }
}

function extractFirstYouTubeUrl(text) {
  const matches = (text || "").match(/https?:\/\/(?:www\.)?(?:youtube\.com\/\S+|youtu\.be\/\S+)/i);
  return matches ? normalizeUrlCandidate(matches[0]) : null;
}

function extractTimestampFromYouTubeUrl(url) {
  try {
    const parsed = new URL(normalizeUrlCandidate(url));
    const tParam = parsed.searchParams.get("t");
    const startParam = parsed.searchParams.get("start");
    return parseYouTubeTimestamp(tParam) ?? parseYouTubeTimestamp(startParam);
  } catch (e) {
    return null;
  }
}

function seekYouTubeToTimestamp(timestampSec) {
  const adjustedSec = Math.max(0, timestampSec - 2);

  if (youtubePlayer && typeof youtubePlayer.seekTo === "function") {
    youtubePlayer.seekTo(adjustedSec, true);
    youtubePlayer.pauseVideo();
    setTimeout(() => {
      if (youtubePlayer && typeof youtubePlayer.playVideo === "function") {
        youtubePlayer.playVideo();
      }
    }, 1000);
  } else if (floatingVideoIframe && floatingVideoIframe.contentWindow) {
    floatingVideoIframe.contentWindow.postMessage(
      JSON.stringify({ event: "command", func: "seekTo", args: [adjustedSec, true] }),
      "*"
    );
    floatingVideoIframe.contentWindow.postMessage(
      JSON.stringify({ event: "command", func: "pauseVideo", args: [] }),
      "*"
    );
    setTimeout(() => {
      if (floatingVideoIframe && floatingVideoIframe.contentWindow) {
        floatingVideoIframe.contentWindow.postMessage(
          JSON.stringify({ event: "command", func: "playVideo", args: [] }),
          "*"
        );
      }
    }, 1000);
  }
  if (floatingVideoResumeHint) {
    floatingVideoResumeHint.classList.add("hidden");
  }
}

async function searchQdrantAndSeekVideo(query) {
  try {
    const response = await fetch("/api/qdrant-search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    if (!response.ok) return;
    const data = await response.json();
    if (data.error) return;

    const videoResults = data.video_results || [];
    const bestVideo = videoResults.find(
      r => r.youtube_deeplink && r.score >= QDRANT_VIDEO_SCORE_THRESHOLD
    );

    if (!bestVideo) return;

    const { video_id, timestamp_sec, timestamp_hms, youtube_deeplink } = bestVideo;
    const currentVideoId = extractYouTubeVideoId(currentYouTubeWatchUrl);

    if (currentVideoId === video_id && floatingVideoContainer) {
      // Same video is playing — seek silently to the relevant timestamp
      seekYouTubeToTimestamp(timestamp_sec);
      currentYouTubeWatchUrl = youtube_deeplink;
    } else {
      // Different video or no video loaded yet — play from the timestamp
      playYouTubeVideo(youtube_deeplink);
    }
  } catch (e) {
    console.warn("Qdrant search failed:", e);
  }
}

function isVideoFallbackReply(text) {
  const normalized = normalizeKeywordText(text);
  return normalized.includes("cannot play videos")
    || normalized.includes("cant play videos")
    || normalized.includes("unable to play videos")
    || normalized.includes("unable to play video")
    || normalized.includes("i am unable to play videos")
    || normalized.includes("i am unable to play video");
}

function isResumeVideoCommand(text) {
  const normalized = normalizeKeywordText(text);
  if (!normalized) {
    return false;
  }
  return normalized.includes("please continue to play the video")
    || normalized.includes("continue to play the video")
    || (normalized.includes("continue") && normalized.includes("play") && normalized.includes("video"));
}

function isNoOrSatisfied(text) {
  const normalized = normalizeKeywordText(text);
  if (!normalized) {
    return false;
  }
  return /\b(no|nope|nah|none|nothing)\b/.test(normalized)
    || normalized.includes("no thanks")
    || normalized.includes("no thank you")
    || normalized.includes("no questions")
    || normalized.includes("nothing else")
    || normalized.includes("that is all")
    || normalized.includes("thats all")
    || normalized.includes("i am good")
    || normalized.includes("im good")
    || normalized.includes("i am satisfied")
    || normalized.includes("satisfied");
}

function logConversationMessage(role, text) {
  if (!text || !text.trim()) return;
  conversationLog.push({ role, text: text.trim() });
}

function isUserQuestion(text) {
  if (!text || text.length < 4) return false;
  const t = text.toLowerCase().trim();
  if (t.endsWith("?")) return true;
  return /^(?:what|where|when|how|why|who|which|is |are |do |does |did |will |would |could |can |is there|are there|do you|does it|have you)\b/.test(t);
}

function extractTopicLabel(text) {
  const cleaned = text.trim().replace(/[.!?]+$/, "");
  const patterns = [
    /^(?:i (?:want|would like|like|need) to (?:see|know|learn about|find out about|ask about|ask))(?: (?:about|more about))?(?: (?:the|a|an))?\s*(.*)/i,
    /^(?:i (?:am|'m) (?:interested|curious)(?: in| about))?(?: (?:the|a|an))?\s*(.*)/i,
    /^(?:can you (?:show|tell) me(?: (?:about|the|a|an))?)\s*(.*)/i,
    /^(?:tell me (?:about|more about)(?: (?:the|a|an))?)\s*(.*)/i,
    /^(?:show me(?: (?:the|a|an))?)\s*(.*)/i,
    /^(?:(?:please )?(?:show|describe|explain)(?: (?:the|a|an))?)\s*(.*)/i,
  ];
  for (const pattern of patterns) {
    const match = cleaned.match(pattern);
    if (match && match[1] && match[1].trim().length > 2) {
      const topic = match[1].trim();
      return topic.charAt(0).toUpperCase() + topic.slice(1);
    }
  }
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
}

function generateConversationSummary() {
  const qaPairs = [];
  const interests = [];
  const seenTexts = new Set();

  for (let i = 0; i < conversationLog.length; i++) {
    const entry = conversationLog[i];
    if (entry.role !== "user") continue;
    const key = entry.text.toLowerCase();
    if (seenTexts.has(key)) continue;
    seenTexts.add(key);

    if (isUserQuestion(entry.text)) {
      // Find the closest agent reply after this user message
      const nextAgent = conversationLog.slice(i + 1).find(e => e.role === "agent");
      qaPairs.push({
        question: entry.text,
        answer: nextAgent ? nextAgent.text : null,
      });
    } else if (entry.text.length > 4) {
      interests.push(extractTopicLabel(entry.text));
    }
  }

  return { qaPairs, interests };
}

function clearSuppressAgentOutput() {
  suppressAgentOutputForTurn = false;
  if (suppressAgentOutputResetTimer) {
    clearTimeout(suppressAgentOutputResetTimer);
    suppressAgentOutputResetTimer = null;
  }
}

function suppressAgentOutputTemporarily(durationMs = 5000) {
  suppressAgentOutputForTurn = true;
  if (suppressAgentOutputResetTimer) {
    clearTimeout(suppressAgentOutputResetTimer);
  }
  suppressAgentOutputResetTimer = setTimeout(() => {
    suppressAgentOutputForTurn = false;
    suppressAgentOutputResetTimer = null;
  }, durationMs);
}

function showRegistrationLinkPrompt() {
  if (hasShownRegistrationPrompt) {
    return;
  }
  hasShownRegistrationPrompt = true;
  postTourFollowupActive = false;
  pendingRegistrationLinkAfterTurn = false;

  const prompt = `Reply to the user now with this exact sentence: "${REGISTRATION_PROMPT_TEXT} ${REGISTRATION_URL} . We look forward to seeing you soon!" After this reply, do not ask any further follow-up questions.`;
  sendMessage(prompt, { isInternal: true });
}

function showAgentTextMessage(text) {
  const messageDiv = createMessageBubble(text, false, false);
  messagesDiv.appendChild(messageDiv);
  scrollToBottom();
}

function isWelcomeConfirmation(text) {
  const normalized = normalizeKeywordText(text);
  if (!normalized) {
    return false;
  }

  if (/\b(yes|yeah|yep|yup|sure|ok|okay|ready|affirmative|certainly|of course)\b/.test(normalized)) {
    return true;
  }
  if (normalized.includes("let s begin") || normalized.includes("lets begin")) {
    return true;
  }
  if (normalized.includes("let s start") || normalized.includes("lets start")) {
    return true;
  }
  return normalized === "start" || normalized === "begin";
}

function startVirtualTourIntro() {
  if (pendingVirtualTourFallbackTimer) {
    clearTimeout(pendingVirtualTourFallbackTimer);
    pendingVirtualTourFallbackTimer = null;
  }

  const prompt = `Say this sentence exactly in one response, without adding extra text: "${VIRTUAL_TOUR_INTRO_TEXT}"`;
  sendMessage(prompt, { isInternal: true });
  pendingVirtualTourVideo = true;

  // Safety fallback: if streaming events are incomplete, still start the tour video.
  pendingVirtualTourFallbackTimer = setTimeout(() => {
    if (!pendingVirtualTourVideo) {
      return;
    }
    pendingVirtualTourVideo = false;
    playYouTubeVideo(DEFAULT_YOUTUBE_URL);
  }, 12000);
}

function handleWelcomeConfirmation(text, fromVoice = false) {
  if (!awaitingWelcomeConfirmation || hasTriggeredVirtualTourIntro) {
    return false;
  }
  if (!isWelcomeConfirmation(text)) {
    return false;
  }

  awaitingWelcomeConfirmation = false;
  hasTriggeredVirtualTourIntro = true;

  if (fromVoice) {
    // Suppress model response to the raw confirmation turn, then inject scripted intro.
    suppressAgentOutputTemporarily(8000);
    pendingWelcomeScriptAfterTurn = true;
  } else {
    // TEXT PATH: send scripted intro to agent AND directly play video — don't wait for events.
    const introPrompt = `Say this sentence exactly in one response, without adding extra text: "${VIRTUAL_TOUR_INTRO_TEXT}"`;
    sendMessage(introPrompt, { isInternal: true });
    playYouTubeVideo(DEFAULT_YOUTUBE_URL);
  }

  return true;
}

function renderVideoStagePlaceholder() {
  const stage = document.getElementById("videoStage");
  if (!stage) {
    return;
  }
  stage.innerHTML = '<div class="video-placeholder">Virtual tour will begin shortly.</div>';
}

function ensureFloatingVideoPlayer() {
  if (floatingVideoContainer && floatingVideoIframe) {
    return;
  }

  const stage = document.getElementById("videoStage");
  if (!stage) {
    return;
  }
  stage.innerHTML = "";

  const player = document.createElement("div");
  player.className = "floating-video-player";
  player.id = "floatingVideoPlayer";

  const header = document.createElement("div");
  header.className = "floating-video-header";

  const title = document.createElement("div");
  title.className = "floating-video-title";
  title.textContent = "Virtual Tour";

  const controls = document.createElement("div");
  controls.className = "floating-video-controls";

  const openButton = document.createElement("button");
  openButton.type = "button";
  openButton.className = "floating-video-btn";
  openButton.textContent = "Open in YouTube";
  openButton.addEventListener("click", () => {
    window.open(currentYouTubeWatchUrl, "_blank", "noopener,noreferrer");
  });

  controls.appendChild(openButton);
  header.appendChild(title);
  header.appendChild(controls);

  const body = document.createElement("div");
  body.className = "floating-video-body";

  const iframe = document.createElement("iframe");
  iframe.className = "floating-video-iframe";
  iframe.id = "tourVideoIframe";
  iframe.src = "";
  iframe.title = "YouTube video player";
  iframe.allow = "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share";
  iframe.referrerPolicy = "strict-origin-when-cross-origin";
  iframe.allowFullscreen = true;

  const resumeHint = document.createElement("div");
  resumeHint.className = "video-resume-hint hidden";
  resumeHint.textContent = RESUME_VIDEO_HINT_TEXT;

  body.appendChild(iframe);
  body.appendChild(resumeHint);
  player.appendChild(header);
  player.appendChild(body);
  stage.appendChild(player);

  floatingVideoContainer = player;
  floatingVideoIframe = iframe;
  floatingVideoResumeHint = resumeHint;
}

function ensureYouTubeApiReady() {
  if (window.YT && window.YT.Player) {
    return Promise.resolve(window.YT);
  }

  if (youtubeApiReadyPromise) {
    return youtubeApiReadyPromise;
  }

  youtubeApiReadyPromise = new Promise((resolve, reject) => {
    const existingScript = document.querySelector('script[src="https://www.youtube.com/iframe_api"]');
    const previousReadyHandler = window.onYouTubeIframeAPIReady;
    let timeoutId = null;

    window.onYouTubeIframeAPIReady = () => {
      if (typeof previousReadyHandler === "function") {
        previousReadyHandler();
      }
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
      resolve(window.YT);
    };

    if (!existingScript) {
      const tag = document.createElement("script");
      tag.src = "https://www.youtube.com/iframe_api";
      tag.onerror = () => {
        if (timeoutId) {
          clearTimeout(timeoutId);
        }
        reject(new Error("Failed to load YouTube iframe API"));
      };
      document.head.appendChild(tag);
    }

    timeoutId = setTimeout(() => {
      reject(new Error("Timed out waiting for YouTube iframe API"));
    }, 10000);
  });

  return youtubeApiReadyPromise;
}

function onYouTubePlayerStateChange(event) {
  // 0 = ended
  if (event && event.data === 0) {
    handleTourVideoEnded();
    return;
  }

  // Manual scrub to end can report paused/cued without sending state=0.
  // 2 = paused, 5 = cued
  if (event && (event.data === 2 || event.data === 5)) {
    maybeHandleTourEndByProgress();
  }
}

function maybeHandleTourEndByProgress() {
  if (!youtubePlayer || hasHandledTourEnd) {
    return;
  }
  if (typeof youtubePlayer.getCurrentTime !== "function" || typeof youtubePlayer.getDuration !== "function") {
    return;
  }

  const duration = youtubePlayer.getDuration();
  const currentTime = youtubePlayer.getCurrentTime();
  if (!duration || duration <= 0) {
    return;
  }

  const progress = currentTime / duration;
  const remainingSeconds = duration - currentTime;
  const nearEnd = progress >= 0.98 || remainingSeconds <= 2;
  if (nearEnd) {
    handleTourVideoEnded();
  }
}

function startYouTubeEndMonitor() {
  if (youtubeEndMonitorInterval) {
    clearInterval(youtubeEndMonitorInterval);
  }

  youtubeEndMonitorInterval = setInterval(() => {
    if (!youtubePlayer || typeof youtubePlayer.getPlayerState !== "function") {
      return;
    }

    if (hasHandledTourEnd) {
      return;
    }

    if (
      typeof youtubePlayer.getCurrentTime === "function"
      && typeof youtubePlayer.getDuration === "function"
    ) {
      const duration = youtubePlayer.getDuration();
      const currentTime = youtubePlayer.getCurrentTime();
      if (duration > 0 && duration - currentTime <= 0.75) {
        handleTourVideoEnded();
        return;
      }
    }

    const state = youtubePlayer.getPlayerState();
    if (state === 0) {
      handleTourVideoEnded();
      return;
    }

    // 1 = playing. For all other states, check if user is effectively at the end.
    if (state !== 1) {
      maybeHandleTourEndByProgress();
    }
  }, 1000);
}

function onYouTubePlayerReady(event) {
  try {
    if (event && event.target) {
      if (typeof event.target.mute === "function") {
        event.target.mute();
      }
      if (typeof event.target.playVideo === "function") {
        event.target.playVideo();
      }
    }
  } catch (error) {
    console.warn("Unable to autoplay YouTube player:", error);
  }
  startYouTubeEndMonitor();
}

function loadVideoWithYouTubeApi(videoId, startSeconds = null) {
  if (!floatingVideoIframe) {
    return;
  }

  ensureYouTubeApiReady()
    .then(() => {
      if (!floatingVideoIframe) {
        return;
      }

      if (!youtubePlayer) {
        const playerVars = {
          autoplay: 1,
          rel: 0,
          enablejsapi: 1,
          playsinline: 1,
          mute: 1,
          origin: window.location.origin,
        };
        if (startSeconds !== null) {
          playerVars.start = startSeconds;
        }
        youtubePlayer = new window.YT.Player(floatingVideoIframe.id, {
          videoId,
          playerVars,
          events: {
            onStateChange: onYouTubePlayerStateChange,
            onReady: onYouTubePlayerReady,
          },
        });
        return;
      }

      if (startSeconds !== null) {
        youtubePlayer.loadVideoById({ videoId, startSeconds });
      } else {
        youtubePlayer.loadVideoById(videoId);
      }
      if (typeof youtubePlayer.mute === "function") {
        youtubePlayer.mute();
      }
      youtubePlayer.playVideo();
      startYouTubeEndMonitor();
    })
    .catch((error) => {
      console.warn("YouTube API load failed, falling back to iframe src.", error);
      if (floatingVideoIframe) {
        const startParam = startSeconds !== null ? `&start=${startSeconds}` : "";
        floatingVideoIframe.src = `https://www.youtube.com/embed/${videoId}?autoplay=1&rel=0&enablejsapi=1&playsinline=1&mute=1${startParam}`;
      }
    });
}

function handleTourVideoEnded() {
  if (hasHandledTourEnd) {
    return;
  }
  clearSuppressAgentOutput();
  hasHandledTourEnd = true;
  postTourFollowupActive = true;
  hasShownRegistrationPrompt = false;

  const prompt = `The virtual tour has ended. Reply to the user now with this exact sentence: "${TOUR_END_QUESTION}".`;
  sendMessage(prompt, { isInternal: true });
}

function onYouTubeIframeMessage(event) {
  if (!event || typeof event.origin !== "string" || !event.origin.includes("youtube.com")) {
    return;
  }

  let payload = event.data;
  if (typeof payload === "string") {
    try {
      payload = JSON.parse(payload);
    } catch (e) {
      return;
    }
  }
  if (!payload) {
    return;
  }

  let playerState = null;
  if (payload.event === "onStateChange" && typeof payload.info === "number") {
    playerState = payload.info;
  } else if (
    payload.event === "infoDelivery"
    && payload.info
    && typeof payload.info.playerState === "number"
  ) {
    playerState = payload.info.playerState;
  }

  // 0 = ended
  if (playerState === 0) {
    handleTourVideoEnded();
    return;
  }

  // 2 = paused, 5 = cued
  if (playerState === 2 || playerState === 5) {
    maybeHandleTourEndByProgress();
  }
}

window.addEventListener("message", onYouTubeIframeMessage);

function playYouTubeVideo(videoUrl = DEFAULT_YOUTUBE_URL) {
  if (pendingVirtualTourFallbackTimer) {
    clearTimeout(pendingVirtualTourFallbackTimer);
    pendingVirtualTourFallbackTimer = null;
  }

  const now = Date.now();
  if (now - lastVideoCommandAt < VIDEO_COMMAND_COOLDOWN_MS) {
    return;
  }
  lastVideoCommandAt = now;
  suppressVideoFallbackUntil = now + VIDEO_FALLBACK_SUPPRESS_MS;
  hasHandledTourEnd = false;

  const normalizedVideoUrl = normalizeUrlCandidate(videoUrl);
  const videoId = extractYouTubeVideoId(normalizedVideoUrl);
  if (!videoId) {
    addSystemMessage("Unable to play video: invalid YouTube link.");
    return;
  }

  const timestampSec = extractTimestampFromYouTubeUrl(normalizedVideoUrl);
  const currentVideoId = extractYouTubeVideoId(currentYouTubeWatchUrl);

  // If the same video is already loaded, seek to the timestamp instead of reloading.
  if (currentVideoId === videoId && floatingVideoContainer && floatingVideoIframe) {
    if (timestampSec !== null) {
      seekYouTubeToTimestamp(timestampSec);
      currentYouTubeWatchUrl = `https://www.youtube.com/watch?v=${videoId}&t=${timestampSec}s`;
    }
    if (floatingVideoResumeHint) {
      floatingVideoResumeHint.classList.add("hidden");
    }
    return;
  }

  currentYouTubeWatchUrl = timestampSec !== null
    ? `https://www.youtube.com/watch?v=${videoId}&t=${timestampSec}s`
    : `https://www.youtube.com/watch?v=${videoId}`;

  ensureFloatingVideoPlayer();

  if (!floatingVideoContainer || !floatingVideoIframe) {
    return;
  }

  const startParam = timestampSec !== null ? `&start=${timestampSec}` : "";
  floatingVideoIframe.src = `https://www.youtube.com/embed/${videoId}?autoplay=1&rel=0&enablejsapi=1&playsinline=1&mute=1${startParam}`;

  loadVideoWithYouTubeApi(videoId, timestampSec);

  // Secondary nudge for browsers that defer autoplay until player is ready.
  setTimeout(() => {
    if (floatingVideoIframe && floatingVideoIframe.contentWindow) {
      floatingVideoIframe.contentWindow.postMessage(
        JSON.stringify({ event: "command", func: "playVideo", args: [] }),
        "*"
      );
    }
  }, 800);

  if (floatingVideoResumeHint) {
    floatingVideoResumeHint.classList.add("hidden");
  }
}

function pauseYouTubeVideo(showHint = true) {
  if (youtubePlayer && typeof youtubePlayer.pauseVideo === "function") {
    youtubePlayer.pauseVideo();
  }
  if (!floatingVideoIframe || !floatingVideoIframe.contentWindow) {
    return;
  }
  floatingVideoIframe.contentWindow.postMessage(
    JSON.stringify({
      event: "command",
      func: "pauseVideo",
      args: []
    }),
    "*"
  );
  if (floatingVideoResumeHint) {
    floatingVideoResumeHint.classList.remove("hidden");
  }
  if (showHint) {
    addSystemMessage(RESUME_VIDEO_HINT_TEXT);
  }
}

function resumeYouTubeVideo() {
  if (youtubePlayer && typeof youtubePlayer.playVideo === "function") {
    youtubePlayer.playVideo();
  }
  if (!floatingVideoIframe || !floatingVideoIframe.contentWindow) {
    return;
  }
  floatingVideoIframe.contentWindow.postMessage(
    JSON.stringify({
      event: "command",
      func: "playVideo",
      args: []
    }),
    "*"
  );
  if (floatingVideoResumeHint) {
    floatingVideoResumeHint.classList.add("hidden");
  }
}

function isEnglishOnlyInput(text) {
  return !/[^\x00-\x7F]/.test(text || "");
}

function notifyEnglishOnlyOnce() {
  const now = Date.now();
  if (now - lastEnglishOnlyNoticeAt < ENGLISH_NOTICE_COOLDOWN_MS) {
    return;
  }
  lastEnglishOnlyNoticeAt = now;
  addSystemMessage(ENGLISH_ONLY_NOTICE);
}

function unlockInterruptionWindow() {
  const now = Date.now();
  if (now - lastKeywordDetectedAt < KEYWORD_COOLDOWN_MS) {
    return;
  }
  lastKeywordDetectedAt = now;
  const wasUnlocked = Date.now() < interruptionUnlockedUntil;
  interruptionUnlockedUntil = Date.now() + INTERRUPTION_UNLOCK_MS;
  if (audioPlayerNode) {
    audioPlayerNode.port.postMessage({ command: "endOfAudio" });
  }
  pauseYouTubeVideo();
  if (websocket && websocket.readyState === WebSocket.OPEN) {
    const wakeReplyMessage = JSON.stringify({
      type: "text",
      text: WAKE_REPLY_TEXT
    });
    websocket.send(wakeReplyMessage);
  }
  if (!wasUnlocked) {
    addSystemMessage(`Interruption unlocked for ${Math.floor(INTERRUPTION_UNLOCK_MS / 1000)} seconds`);
  }
}

function ensureKeywordRecognizer() {
  if (keywordRecognizer) {
    return true;
  }

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    addSystemMessage("Keyword detection is not supported in this browser.");
    return false;
  }

  keywordRecognizer = new SpeechRecognition();
  keywordRecognizer.continuous = true;
  keywordRecognizer.interimResults = true;
  keywordRecognizer.lang = "en-US";

  keywordRecognizer.onresult = (event) => {
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0] ? event.results[i][0].transcript : "";
      if (hasInterruptionKeyword(transcript)) {
        unlockInterruptionWindow();
      }
    }
  };

  keywordRecognizer.onend = () => {
    if (keywordRecognizerActive) {
      try {
        keywordRecognizer.start();
      } catch (e) {
        console.warn("Keyword recognizer restart failed:", e);
      }
    }
  };

  keywordRecognizer.onerror = (event) => {
    const err = event && event.error ? event.error : "unknown";
    if (err !== "no-speech" && err !== "aborted") {
      console.warn("Keyword recognizer error:", err);
    }
  };

  return true;
}

function startKeywordRecognizer() {
  if (!ensureKeywordRecognizer()) {
    return;
  }
  if (keywordRecognizerActive) {
    return;
  }
  keywordRecognizerActive = true;
  try {
    keywordRecognizer.start();
  } catch (e) {
    console.warn("Keyword recognizer start failed:", e);
  }
}

// Console logging functionality
function formatTimestamp() {
  const now = new Date();
  return now.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit', fractionalSecondDigits: 3 });
}

function addConsoleEntry(type, content, data = null, emoji = null, author = null, isAudio = false) {
  // Skip audio events if checkbox is unchecked
  if (isAudio && !showAudioEventsCheckbox.checked) {
    return;
  }

  const entry = document.createElement("div");
  entry.className = `console-entry ${type}`;

  const header = document.createElement("div");
  header.className = "console-entry-header";

  const leftSection = document.createElement("div");
  leftSection.className = "console-entry-left";

  // Add emoji icon if provided
  if (emoji) {
    const emojiIcon = document.createElement("span");
    emojiIcon.className = "console-entry-emoji";
    emojiIcon.textContent = emoji;
    leftSection.appendChild(emojiIcon);
  }

  // Add expand/collapse icon
  const expandIcon = document.createElement("span");
  expandIcon.className = "console-expand-icon";
  expandIcon.textContent = data ? "▶" : "";

  const typeLabel = document.createElement("span");
  typeLabel.className = "console-entry-type";
  typeLabel.textContent = type === 'outgoing' ? '↑ Upstream' : type === 'incoming' ? '↓ Downstream' : '⚠ Error';

  leftSection.appendChild(expandIcon);
  leftSection.appendChild(typeLabel);

  // Add author badge if provided
  if (author) {
    const authorBadge = document.createElement("span");
    authorBadge.className = "console-entry-author";
    authorBadge.textContent = author;
    authorBadge.setAttribute('data-author', author);
    leftSection.appendChild(authorBadge);
  }

  const timestamp = document.createElement("span");
  timestamp.className = "console-entry-timestamp";
  timestamp.textContent = formatTimestamp();

  header.appendChild(leftSection);
  header.appendChild(timestamp);

  const contentDiv = document.createElement("div");
  contentDiv.className = "console-entry-content";
  contentDiv.textContent = content;

  entry.appendChild(header);
  entry.appendChild(contentDiv);

  // JSON details (hidden by default)
  let jsonDiv = null;
  if (data) {
    jsonDiv = document.createElement("div");
    jsonDiv.className = "console-entry-json collapsed";
    const pre = document.createElement("pre");
    pre.textContent = JSON.stringify(data, null, 2);
    jsonDiv.appendChild(pre);
    entry.appendChild(jsonDiv);

    // Make entry clickable if it has data
    entry.classList.add("expandable");

    // Toggle expand/collapse on click
    entry.addEventListener("click", () => {
      const isExpanded = !jsonDiv.classList.contains("collapsed");

      if (isExpanded) {
        // Collapse
        jsonDiv.classList.add("collapsed");
        expandIcon.textContent = "▶";
        entry.classList.remove("expanded");
      } else {
        // Expand
        jsonDiv.classList.remove("collapsed");
        expandIcon.textContent = "▼";
        entry.classList.add("expanded");
      }
    });
  }

  consoleContent.appendChild(entry);
  consoleContent.scrollTop = consoleContent.scrollHeight;
}

function clearConsole() {
  consoleContent.innerHTML = '';
}

function toggleConsolePopup(show) {
  if (!consolePopup) {
    return;
  }
  if (show) {
    consolePopup.classList.remove("hidden-popup");
  } else {
    consolePopup.classList.add("hidden-popup");
  }
}

// Clear console button handler
if (clearConsoleBtn) {
  clearConsoleBtn.addEventListener('click', clearConsole);
}

if (openConsoleButton) {
  openConsoleButton.addEventListener("click", () => toggleConsolePopup(true));
}

if (closeConsoleButton) {
  closeConsoleButton.addEventListener("click", () => toggleConsolePopup(false));
}

// Update connection status UI
function updateConnectionStatus(connected) {
  if (connected) {
    statusIndicator.classList.remove("disconnected");
    statusText.textContent = "Connected";
  } else {
    statusIndicator.classList.add("disconnected");
    statusText.textContent = "Disconnected";
  }
}

// Create a message bubble element
function createProfileIcon(isUser) {
  const avatarDiv = document.createElement("div");
  avatarDiv.className = `message-avatar ${isUser ? "user" : "agent"}`;
  avatarDiv.textContent = isUser ? "U" : "S";
  avatarDiv.setAttribute("aria-hidden", "true");
  return avatarDiv;
}

function setBubbleTextWithLinks(element, text) {
  element.textContent = "";
  const content = text || "";
  const urlRegex = /(https?:\/\/[^\s]+)/gi;
  let lastIndex = 0;
  let match;

  while ((match = urlRegex.exec(content)) !== null) {
    const start = match.index;
    const end = start + match[0].length;
    if (start > lastIndex) {
      element.appendChild(document.createTextNode(content.slice(lastIndex, start)));
    }
    const link = document.createElement("a");
    link.href = match[0];
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = match[0];
    element.appendChild(link);
    lastIndex = end;
  }

  if (lastIndex < content.length) {
    element.appendChild(document.createTextNode(content.slice(lastIndex)));
  }
}

function createMessageBubble(text, isUser, isPartial = false) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${isUser ? "user" : "agent"}`;

  const avatarDiv = createProfileIcon(isUser);
  const bubbleDiv = document.createElement("div");
  bubbleDiv.className = "bubble";

  const textP = document.createElement("p");
  textP.className = "bubble-text";
  setBubbleTextWithLinks(textP, text);

  // Add typing indicator for partial messages
  if (isPartial && !isUser) {
    const typingSpan = document.createElement("span");
    typingSpan.className = "typing-indicator";
    textP.appendChild(typingSpan);
  }

  bubbleDiv.appendChild(textP);
  if (isUser) {
    messageDiv.appendChild(bubbleDiv);
    messageDiv.appendChild(avatarDiv);
  } else {
    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(bubbleDiv);
  }

  return messageDiv;
}

// Update existing message bubble text
function updateMessageBubble(element, text, isPartial = false) {
  const textElement = element.querySelector(".bubble-text");

  // Remove existing typing indicator
  const existingIndicator = textElement.querySelector(".typing-indicator");
  if (existingIndicator) {
    existingIndicator.remove();
  }

  setBubbleTextWithLinks(textElement, text);

  // Add typing indicator for partial messages
  if (isPartial) {
    const typingSpan = document.createElement("span");
    typingSpan.className = "typing-indicator";
    textElement.appendChild(typingSpan);
  }
}

// Add a system message
function addSystemMessage(text) {
  const messageDiv = document.createElement("div");
  messageDiv.className = "system-message";
  messageDiv.textContent = text;
  messagesDiv.appendChild(messageDiv);
  scrollToBottom();
}

function formatDurationLabel(milliseconds) {
  const totalSeconds = Math.max(0, Math.floor(milliseconds / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  if (hours > 0) {
    const hourLabel = hours === 1 ? "hour" : "hours";
    const minuteLabel = minutes === 1 ? "minute" : "minutes";
    const secondLabel = seconds === 1 ? "second" : "seconds";
    return `${hours} ${hourLabel} ${minutes} ${minuteLabel} ${seconds} ${secondLabel}`;
  }
  if (minutes === 0) {
    const secondLabel = seconds === 1 ? "second" : "seconds";
    return `${seconds} ${secondLabel}`;
  }
  const minuteLabel = minutes === 1 ? "minute" : "minutes";
  const secondLabel = seconds === 1 ? "second" : "seconds";
  return `${minutes} ${minuteLabel} ${seconds} ${secondLabel}`;
}

function markConversationStarted() {
  if (!conversationStartedAt) {
    conversationStartedAt = Date.now();
  }
}

function showEndConversationSummary() {
  const startAt = conversationStartedAt || Date.now();
  const endAt = sessionEndedAt || Date.now();
  const durationMs = Math.max(0, endAt - startAt);

  if (summarySessionId) {
    summarySessionId.textContent = sessionId;
  }
  if (summaryUserName) {
    summaryUserName.textContent = userInfo?.fullName || "-";
  }
  if (summaryUserEmail) {
    summaryUserEmail.textContent = userInfo?.emailAddress || "-";
  }
  if (summaryDuration) {
    summaryDuration.textContent = formatDurationLabel(durationMs);
  }

  // Populate interests and questions from conversation log
  const { qaPairs, interests } = generateConversationSummary();

  const summaryInterestsList = document.getElementById("summaryInterestsList");
  if (summaryInterestsList) {
    summaryInterestsList.innerHTML = "";
    if (interests.length > 0) {
      interests.forEach(item => {
        const li = document.createElement("li");
        li.textContent = item;
        summaryInterestsList.appendChild(li);
      });
    } else {
      summaryInterestsList.innerHTML = "<li class='summary-empty'>No specific topics recorded.</li>";
    }
  }

  const summaryQuestionsList = document.getElementById("summaryQuestionsList");
  if (summaryQuestionsList) {
    summaryQuestionsList.innerHTML = "";
    if (qaPairs.length > 0) {
      qaPairs.forEach(({ question, answer }) => {
        const li = document.createElement("li");
        li.className = "summary-qa-item";

        const qDiv = document.createElement("div");
        qDiv.className = "summary-qa-question";
        qDiv.textContent = question;

        li.appendChild(qDiv);

        if (answer) {
          const aDiv = document.createElement("div");
          aDiv.className = "summary-qa-answer";
          const truncated = answer.length > 220 ? answer.slice(0, 220).trimEnd() + "…" : answer;
          aDiv.textContent = truncated;
          li.appendChild(aDiv);
        }

        summaryQuestionsList.appendChild(li);
      });
    } else {
      summaryQuestionsList.innerHTML = "<li class='summary-empty'>No questions recorded.</li>";
    }
  }

  // Wire up share button — sends directly to Star Learners team WhatsApp
  const shareBtn = document.getElementById("summaryShareButton");
  if (shareBtn) {
    shareBtn.onclick = () => {
      const name = userInfo?.fullName || "a visitor";
      const topicsText = interests.length > 0
        ? interests.map(t => `  • ${t}`).join("\n")
        : "  • General virtual tour";
      const qaText = qaPairs.length > 0
        ? qaPairs.map(({ question, answer }) => {
            const a = answer ? `\n    ↳ ${answer.slice(0, 160).trimEnd()}${answer.length > 160 ? "…" : ""}` : "";
            return `  • ${question}${a}`;
          }).join("\n")
        : "  • None";
      const msg = [
        `👋 Hi! I'm ${name} and I just completed a virtual tour with STELLA at Star Learners.`,
        "",
        `📧 ${userInfo?.emailAddress || ""}`,
        "",
        "📌 Topics I explored:",
        topicsText,
        "",
        "❓ My questions:",
        qaText,
        "",
        "I'd love to learn more — please get in touch!",
      ].join("\n");
      window.open(`https://wa.me/${TEAM_WHATSAPP_NUMBER}?text=${encodeURIComponent(msg)}`, "_blank", "noopener,noreferrer");
    };
  }

  // Wire up connect button
  const connectBtn = document.getElementById("summaryConnectButton");
  if (connectBtn) {
    connectBtn.onclick = () => {
      window.open(TEAM_CONNECT_URL, "_blank", "noopener,noreferrer");
    };
  }

  if (endConversationModal) {
    endConversationModal.classList.add("show");
  }
}

function stopKeywordRecognizer() {
  keywordRecognizerActive = false;
  if (keywordRecognizer) {
    try {
      keywordRecognizer.stop();
    } catch (e) {
      console.warn("Keyword recognizer stop failed:", e);
    }
  }
}

function stopAudioSession() {
  setInputMode("text", true);
  if (audioPlayerNode) {
    audioPlayerNode.port.postMessage({ command: "endOfAudio" });
  }
}

function endConversation() {
  if (hasEndedConversation || isEndingConversation) {
    return;
  }
  isEndingConversation = true;
  hasEndedConversation = true;
  sessionEndedAt = Date.now();

  stopAudioSession();
  clearSuppressAgentOutput();

  if (startAudioButton) {
    startAudioButton.disabled = true;
  }
  if (endConversationButton) {
    endConversationButton.disabled = true;
  }

  const canClose = websocket && (
    websocket.readyState === WebSocket.OPEN
    || websocket.readyState === WebSocket.CONNECTING
  );
  if (canClose) {
    websocket.close(1000, "Conversation ended by user");
  } else {
    showEndConversationSummary();
  }
}

// Scroll to bottom of messages
function scrollToBottom() {
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// Sanitize event data for console display (replace large audio data with summary)
function sanitizeEventForDisplay(event) {
  // Deep clone the event object
  const sanitized = JSON.parse(JSON.stringify(event));

  // Check for audio data in content.parts
  if (sanitized.content && sanitized.content.parts) {
    sanitized.content.parts = sanitized.content.parts.map(part => {
      if (part.inlineData && part.inlineData.data) {
        // Calculate byte size (base64 string length / 4 * 3, roughly)
        const byteSize = Math.floor(part.inlineData.data.length * 0.75);
        return {
          ...part,
          inlineData: {
            ...part.inlineData,
            data: `(${byteSize.toLocaleString()} bytes)`
          }
        };
      }
      return part;
    });
  }

  return sanitized;
}

// WebSocket handlers
function connectWebsocket() {
  if (hasEndedConversation) {
    return;
  }

  // Connect websocket
  const ws_url = getWebSocketUrl();
  websocket = new WebSocket(ws_url);

  // Handle connection open
  websocket.onopen = function () {
    if (hasEndedConversation) {
      websocket.close(1000, "Conversation already ended");
      return;
    }

    console.log("WebSocket connection opened.");
    updateConnectionStatus(true);
    addSystemMessage("Connected to ADK streaming server");

    // Log to console
    addConsoleEntry('incoming', 'WebSocket Connected', {
      userId: userId,
      sessionId: sessionId,
      url: ws_url
    }, '🔌', 'system');

    updateSendButtonState();
    requestAgentWelcome();
  };

  // Handle incoming messages
  websocket.onmessage = function (event) {
    if (hasEndedConversation) {
      return;
    }

    // Parse the incoming ADK Event
    const adkEvent = JSON.parse(event.data);
    console.log("[AGENT TO CLIENT] ", adkEvent);

    // Log to console panel
    let eventSummary = 'Event';
    let eventEmoji = '📨'; // Default emoji
    const author = adkEvent.author || 'system';

    if (adkEvent.turnComplete) {
      eventSummary = 'Turn Complete';
      eventEmoji = '✅';
    } else if (adkEvent.interrupted) {
      eventSummary = 'Interrupted';
      eventEmoji = '⏸️';
    } else if (adkEvent.inputTranscription) {
      // Show transcription text in summary
      const transcriptionText = adkEvent.inputTranscription.text || '';
      const truncated = transcriptionText.length > 60
        ? transcriptionText.substring(0, 60) + '...'
        : transcriptionText;
      eventSummary = `Input Transcription: "${truncated}"`;
      eventEmoji = '📝';
    } else if (adkEvent.outputTranscription) {
      // Show transcription text in summary
      const transcriptionText = adkEvent.outputTranscription.text || '';
      const truncated = transcriptionText.length > 60
        ? transcriptionText.substring(0, 60) + '...'
        : transcriptionText;
      eventSummary = `Output Transcription: "${truncated}"`;
      eventEmoji = '📝';
    } else if (adkEvent.usageMetadata) {
      // Show token usage information
      const usage = adkEvent.usageMetadata;
      const promptTokens = usage.promptTokenCount || 0;
      const responseTokens = usage.candidatesTokenCount || 0;
      const totalTokens = usage.totalTokenCount || 0;
      eventSummary = `Token Usage: ${totalTokens.toLocaleString()} total (${promptTokens.toLocaleString()} prompt + ${responseTokens.toLocaleString()} response)`;
      eventEmoji = '📊';
    } else if (adkEvent.content && adkEvent.content.parts) {
      const hasText = adkEvent.content.parts.some(p => p.text);
      const hasAudio = adkEvent.content.parts.some(p => p.inlineData);
      const hasExecutableCode = adkEvent.content.parts.some(p => p.executableCode);
      const hasCodeExecutionResult = adkEvent.content.parts.some(p => p.codeExecutionResult);

      if (hasExecutableCode) {
        // Show executable code
        const codePart = adkEvent.content.parts.find(p => p.executableCode);
        if (codePart && codePart.executableCode) {
          const code = codePart.executableCode.code || '';
          const language = codePart.executableCode.language || 'unknown';
          const truncated = code.length > 60
            ? code.substring(0, 60).replace(/\n/g, ' ') + '...'
            : code.replace(/\n/g, ' ');
          eventSummary = `Executable Code (${language}): ${truncated}`;
          eventEmoji = '💻';
        }
      }

      if (hasCodeExecutionResult) {
        // Show code execution result
        const resultPart = adkEvent.content.parts.find(p => p.codeExecutionResult);
        if (resultPart && resultPart.codeExecutionResult) {
          const outcome = resultPart.codeExecutionResult.outcome || 'UNKNOWN';
          const output = resultPart.codeExecutionResult.output || '';
          const truncatedOutput = output.length > 60
            ? output.substring(0, 60).replace(/\n/g, ' ') + '...'
            : output.replace(/\n/g, ' ');
          eventSummary = `Code Execution Result (${outcome}): ${truncatedOutput}`;
          eventEmoji = outcome === 'OUTCOME_OK' ? '✅' : '❌';
        }
      }

      if (hasText) {
        // Show text preview in summary
        const textPart = adkEvent.content.parts.find(p => p.text);
        if (textPart && textPart.text) {
          const text = textPart.text;
          const truncated = text.length > 80
            ? text.substring(0, 80) + '...'
            : text;
          eventSummary = `Text: "${truncated}"`;
          eventEmoji = '💭';
        } else {
          eventSummary = 'Text Response';
          eventEmoji = '💭';
        }
      }

      if (hasAudio) {
        // Extract audio info for summary
        const audioPart = adkEvent.content.parts.find(p => p.inlineData);
        if (audioPart && audioPart.inlineData) {
          const mimeType = audioPart.inlineData.mimeType || 'unknown';
          const dataLength = audioPart.inlineData.data ? audioPart.inlineData.data.length : 0;
          // Base64 string length / 4 * 3 gives approximate bytes
          const byteSize = Math.floor(dataLength * 0.75);
          eventSummary = `Audio Response: ${mimeType} (${byteSize.toLocaleString()} bytes)`;
          eventEmoji = '🔊';
        } else {
          eventSummary = 'Audio Response';
          eventEmoji = '🔊';
        }

        // Log audio event with isAudio flag (filtered by checkbox)
        const sanitizedEvent = sanitizeEventForDisplay(adkEvent);
        addConsoleEntry('incoming', eventSummary, sanitizedEvent, eventEmoji, author, true);
      }
    }

    // Create a sanitized version for console display (replace large audio data with summary)
    // Skip if already logged as audio event above
    const isAudioOnlyEvent = adkEvent.content && adkEvent.content.parts &&
      adkEvent.content.parts.some(p => p.inlineData) &&
      !adkEvent.content.parts.some(p => p.text);
    if (!isAudioOnlyEvent) {
      const sanitizedEvent = sanitizeEventForDisplay(adkEvent);
      addConsoleEntry('incoming', eventSummary, sanitizedEvent, eventEmoji, author);
    }

    // Handle turn complete event
    if (adkEvent.turnComplete === true) {
      isAgentSpeaking = false;
      interruptionUnlockedUntil = 0;
      clearSuppressAgentOutput();
      // Remove typing indicator from current message
      if (currentBubbleElement) {
        const textElement = currentBubbleElement.querySelector(".bubble-text");
        const typingIndicator = textElement.querySelector(".typing-indicator");
        if (typingIndicator) {
          typingIndicator.remove();
        }
      }
      // Remove typing indicator from current output transcription
      if (currentOutputTranscriptionElement) {
        const textElement = currentOutputTranscriptionElement.querySelector(".bubble-text");
        const typingIndicator = textElement.querySelector(".typing-indicator");
        if (typingIndicator) {
          typingIndicator.remove();
        }
      }
      currentMessageId = null;
      currentBubbleElement = null;
      currentOutputTranscriptionId = null;
      currentOutputTranscriptionElement = null;
      inputTranscriptionFinished = false; // Reset for next turn
      hasOutputTranscriptionInTurn = false; // Reset for next turn

      if (pendingWelcomeScriptAfterTurn) {
        pendingWelcomeScriptAfterTurn = false;
        startVirtualTourIntro();
      } else if (pendingVirtualTourVideo) {
        // Fallback path: start video on turn complete even if output transcription stream is partial.
        pendingVirtualTourVideo = false;
        if (pendingVirtualTourFallbackTimer) {
          clearTimeout(pendingVirtualTourFallbackTimer);
          pendingVirtualTourFallbackTimer = null;
        }
        playYouTubeVideo(DEFAULT_YOUTUBE_URL);
      } else if (pendingRegistrationLinkAfterTurn) {
        pendingRegistrationLinkAfterTurn = false;
        showRegistrationLinkPrompt();
      }

      return;
    }

    // Handle interrupted event
    if (adkEvent.interrupted === true) {
      isAgentSpeaking = false;
      interruptionUnlockedUntil = 0;
      clearSuppressAgentOutput();
      // Stop audio playback if it's playing
      if (audioPlayerNode) {
        audioPlayerNode.port.postMessage({ command: "endOfAudio" });
      }
      pauseYouTubeVideo(!hasHandledTourEnd);

      // Keep the partial message but mark it as interrupted
      if (currentBubbleElement) {
        const textElement = currentBubbleElement.querySelector(".bubble-text");

        // Remove typing indicator
        const typingIndicator = textElement.querySelector(".typing-indicator");
        if (typingIndicator) {
          typingIndicator.remove();
        }

        // Add interrupted marker
        currentBubbleElement.classList.add("interrupted");
      }

      // Keep the partial output transcription but mark it as interrupted
      if (currentOutputTranscriptionElement) {
        const textElement = currentOutputTranscriptionElement.querySelector(".bubble-text");

        // Remove typing indicator
        const typingIndicator = textElement.querySelector(".typing-indicator");
        if (typingIndicator) {
          typingIndicator.remove();
        }

        // Add interrupted marker
        currentOutputTranscriptionElement.classList.add("interrupted");
      }

      // Reset state so new content creates a new bubble
      currentMessageId = null;
      currentBubbleElement = null;
      currentOutputTranscriptionId = null;
      currentOutputTranscriptionElement = null;
      inputTranscriptionFinished = false; // Reset for next turn
      hasOutputTranscriptionInTurn = false; // Reset for next turn
      return;
    }

    // Handle input transcription (user's spoken words)
    if (adkEvent.inputTranscription && adkEvent.inputTranscription.text) {
      const transcriptionText = adkEvent.inputTranscription.text;
      const isFinished = adkEvent.inputTranscription.finished;

      if (transcriptionText) {
        if (isFinished) {
          if (isResumeVideoCommand(transcriptionText)) {
            resumeYouTubeVideo();
            suppressAgentOutputTemporarily(3000);
            return;
          }
          handleWelcomeConfirmation(transcriptionText, true);
          logConversationMessage("user", transcriptionText);

          if (postTourFollowupActive && isNoOrSatisfied(transcriptionText)) {
            suppressAgentOutputTemporarily(5000);
            pendingRegistrationLinkAfterTurn = true;
            if (audioPlayerNode) {
              audioPlayerNode.port.postMessage({ command: "endOfAudio" });
            }
            return;
          }

          // Query Qdrant in the background to seek video to the relevant timestamp.
          searchQdrantAndSeekVideo(transcriptionText);
        }

        if (!isEnglishOnlyInput(transcriptionText)) {
          notifyEnglishOnlyOnce();
          return;
        }

        // Ignore late-arriving transcriptions after we've finished for this turn
        if (inputTranscriptionFinished) {
          return;
        }

        if (currentInputTranscriptionId == null) {
          // Create new transcription bubble
          currentInputTranscriptionId = Math.random().toString(36).substring(7);
          // Clean spaces between CJK characters
          const cleanedText = cleanCJKSpaces(transcriptionText);
          currentInputTranscriptionElement = createMessageBubble(cleanedText, true, !isFinished);
          currentInputTranscriptionElement.id = currentInputTranscriptionId;

          // Add a special class to indicate it's a transcription
          currentInputTranscriptionElement.classList.add("transcription");

          messagesDiv.appendChild(currentInputTranscriptionElement);
        } else {
          // Update existing transcription bubble only if model hasn't started responding
          // This prevents late partial transcriptions from overwriting complete ones
          if (currentOutputTranscriptionId == null && currentMessageId == null) {
            if (isFinished) {
              // Final transcription contains the complete text, replace entirely
              const cleanedText = cleanCJKSpaces(transcriptionText);
              updateMessageBubble(currentInputTranscriptionElement, cleanedText, false);
            } else {
              // Partial transcription - append to existing text
              const existingText = currentInputTranscriptionElement.querySelector(".bubble-text").textContent;
              // Remove typing indicator if present
              const cleanText = existingText.replace(/\.\.\.$/, '');
              // Clean spaces between CJK characters before updating
              const accumulatedText = cleanCJKSpaces(cleanText + transcriptionText);
              updateMessageBubble(currentInputTranscriptionElement, accumulatedText, true);
            }
          }
        }

        // If transcription is finished, reset the state and mark as complete
        if (isFinished) {
          currentInputTranscriptionId = null;
          currentInputTranscriptionElement = null;
          inputTranscriptionFinished = true; // Prevent duplicate bubbles from late events
        }

        scrollToBottom();
      }
    }

    // Handle output transcription (model's spoken words)
    if (adkEvent.outputTranscription && adkEvent.outputTranscription.text) {
      const transcriptionText = adkEvent.outputTranscription.text;
      const isFinished = adkEvent.outputTranscription.finished;
      markConversationStarted();
      hasOutputTranscriptionInTurn = true;
      isAgentSpeaking = true;

      if (transcriptionText) {
        if (suppressAgentOutputForTurn) {
          if (audioPlayerNode) {
            audioPlayerNode.port.postMessage({ command: "endOfAudio" });
          }
          return;
        }

        if (Date.now() < suppressVideoFallbackUntil && isVideoFallbackReply(transcriptionText)) {
          if (audioPlayerNode) {
            audioPlayerNode.port.postMessage({ command: "endOfAudio" });
          }
          return;
        }

        // Finalize any active input transcription when server starts responding
        if (currentInputTranscriptionId != null && currentOutputTranscriptionId == null) {
          // This is the first output transcription - finalize input transcription
          const textElement = currentInputTranscriptionElement.querySelector(".bubble-text");
          const typingIndicator = textElement.querySelector(".typing-indicator");
          if (typingIndicator) {
            typingIndicator.remove();
          }
          // Reset input transcription state so next user input creates new balloon
          currentInputTranscriptionId = null;
          currentInputTranscriptionElement = null;
          inputTranscriptionFinished = true; // Prevent duplicate bubbles from late events
        }

        if (currentOutputTranscriptionId == null) {
          // Create new transcription bubble for agent
          currentOutputTranscriptionId = Math.random().toString(36).substring(7);
          currentOutputTranscriptionElement = createMessageBubble(transcriptionText, false, !isFinished);
          currentOutputTranscriptionElement.id = currentOutputTranscriptionId;

          // Add a special class to indicate it's a transcription
          currentOutputTranscriptionElement.classList.add("transcription");

          messagesDiv.appendChild(currentOutputTranscriptionElement);
        } else {
          // Update existing transcription bubble
          if (isFinished) {
            // Final transcription contains the complete text, replace entirely
            updateMessageBubble(currentOutputTranscriptionElement, transcriptionText, false);
          } else {
            // Partial transcription - append to existing text
            const existingText = currentOutputTranscriptionElement.querySelector(".bubble-text").textContent;
            // Remove typing indicator if present
            const cleanText = existingText.replace(/\.\.\.$/, '');
            updateMessageBubble(currentOutputTranscriptionElement, cleanText + transcriptionText, true);
          }
        }

        // If transcription is finished, reset the state
        if (isFinished) {
          logConversationMessage("agent", transcriptionText);
          if (pendingVirtualTourVideo) {
            pendingVirtualTourVideo = false;
            if (pendingVirtualTourFallbackTimer) {
              clearTimeout(pendingVirtualTourFallbackTimer);
              pendingVirtualTourFallbackTimer = null;
            }
            playYouTubeVideo(DEFAULT_YOUTUBE_URL);
          }
          currentOutputTranscriptionId = null;
          currentOutputTranscriptionElement = null;
        }

        scrollToBottom();
      }
    }

    // Handle content events (text or audio)
    if (adkEvent.content && adkEvent.content.parts) {
      const parts = adkEvent.content.parts;

      if (suppressAgentOutputForTurn) {
        if (audioPlayerNode) {
          audioPlayerNode.port.postMessage({ command: "endOfAudio" });
        }
        return;
      }

      // Finalize any active input transcription when server starts responding with content
      if (currentInputTranscriptionId != null && currentMessageId == null && currentOutputTranscriptionId == null) {
        // This is the first content event - finalize input transcription
        const textElement = currentInputTranscriptionElement.querySelector(".bubble-text");
        const typingIndicator = textElement.querySelector(".typing-indicator");
        if (typingIndicator) {
          typingIndicator.remove();
        }
        // Reset input transcription state so next user input creates new balloon
        currentInputTranscriptionId = null;
        currentInputTranscriptionElement = null;
        inputTranscriptionFinished = true; // Prevent duplicate bubbles from late events
      }

      for (const part of parts) {
        // Handle inline data (audio)
        if (part.inlineData) {
          const mimeType = part.inlineData.mimeType;
          const data = part.inlineData.data;

          if (mimeType && mimeType.startsWith("audio/pcm") && audioPlayerNode) {
            markConversationStarted();
            isAgentSpeaking = true;
            audioPlayerNode.port.postMessage(base64ToArray(data));
          }
        }

        // Handle text
        if (part.text) {
          if (Date.now() < suppressVideoFallbackUntil && isVideoFallbackReply(part.text)) {
            if (audioPlayerNode) {
              audioPlayerNode.port.postMessage({ command: "endOfAudio" });
            }
            continue;
          }

          const youtubeUrlInText = extractFirstYouTubeUrl(part.text);
          if (youtubeUrlInText) {
            const urlTimestamp = extractTimestampFromYouTubeUrl(youtubeUrlInText);
            // Only auto-play from agent text if: no timestamp (fresh load), or tour already active
            if (urlTimestamp === null || hasTriggeredVirtualTourIntro) {
              playYouTubeVideo(youtubeUrlInText);
            }
          }

          // Skip thinking/reasoning text from chat bubbles (shown in event console)
          if (part.thought) {
            continue;
          }

          // Skip final aggregated content when output transcription already
          // delivered the response (prevents duplicate thinking text replay)
          if (!adkEvent.partial && hasOutputTranscriptionInTurn) {
            continue;
          }

          // Add a new message bubble for a new turn
          if (currentMessageId == null) {
            currentMessageId = Math.random().toString(36).substring(7);
            currentBubbleElement = createMessageBubble(part.text, false, true);
            currentBubbleElement.id = currentMessageId;
            messagesDiv.appendChild(currentBubbleElement);
          } else {
            // Update the existing message bubble with accumulated text
            const existingText = currentBubbleElement.querySelector(".bubble-text").textContent;
            // Remove the "..." if present
            const cleanText = existingText.replace(/\.\.\.$/, '');
            updateMessageBubble(currentBubbleElement, cleanText + part.text, true);
          }

          // Scroll down to the bottom of the messagesDiv
          scrollToBottom();
        }
      }
    }
  };

  // Handle connection close
  websocket.onclose = function () {
    console.log("WebSocket connection closed.");
    updateConnectionStatus(false);
    updateSendButtonState();
    if (hasEndedConversation || isEndingConversation) {
      addSystemMessage("Conversation ended.");
      showEndConversationSummary();
      isEndingConversation = false;
      return;
    }

    addSystemMessage("Connection closed. Reconnecting in 5 seconds...");

    // Log to console
    addConsoleEntry('error', 'WebSocket Disconnected', {
      status: 'Connection closed',
      reconnecting: true,
      reconnectDelay: '5 seconds'
    }, '🔌', 'system');

    setTimeout(function () {
      console.log("Reconnecting...");

      // Log reconnection attempt to console
      addConsoleEntry('outgoing', 'Reconnecting to ADK server...', {
        userId: userId,
        sessionId: sessionId
      }, '🔄', 'system');

      connectWebsocket();
    }, 5000);
  };

  websocket.onerror = function (e) {
    console.log("WebSocket error: ", e);
    updateConnectionStatus(false);

    // Log to console
    addConsoleEntry('error', 'WebSocket Error', {
      error: e.type,
      message: 'Connection error occurred'
    }, '⚠️', 'system');
  };
}


// Send a message to the server as JSON
function sendMessage(message, options = {}) {
  if (websocket && websocket.readyState == WebSocket.OPEN) {
    const isInternal = options.isInternal === true;
    if (!isInternal && !isEnglishOnlyInput(message)) {
      notifyEnglishOnlyOnce();
      return;
    }

    const jsonMessage = JSON.stringify({
      type: "text",
      text: message
    });
    websocket.send(jsonMessage);

    // Log to console panel
    addConsoleEntry(
      'outgoing',
      (isInternal ? 'Internal Prompt: ' : 'User Message: ') + message,
      null,
      isInternal ? '🤖' : '💬',
      isInternal ? 'system' : 'user'
    );
  }
}

// Text input handling
const textInput = document.getElementById("textInput");
const sendTextButton = document.getElementById("sendTextButton");
const DEFAULT_TEXT_INPUT_PLACEHOLDER = "Ask anything";
const VOICE_TEXT_INPUT_PLACEHOLDER = "Voice mode active. Speak to Stella...";
let inputMode = "text";

function updateSendButtonState() {
  if (sendTextButton && textInput) {
    const hasText = !!textInput.value.trim();
    const wsReady = !!(websocket && websocket.readyState === WebSocket.OPEN);
    sendTextButton.disabled = !(hasText && wsReady);
  }
}

function submitTextMessage() {
  if (!textInput) return;
  if (inputMode === "voice" || textInput.disabled) return;
  const text = textInput.value.trim();
  if (!text) return;
  if (!websocket || websocket.readyState !== WebSocket.OPEN) return;

  // Show user bubble immediately
  const bubble = createMessageBubble(text, true, false);
  messagesDiv.appendChild(bubble);
  scrollToBottom();

  // Handle resume video command without forwarding to agent.
  if (isResumeVideoCommand(text)) {
    resumeYouTubeVideo();
    suppressAgentOutputTemporarily(3000);
    textInput.value = "";
    textInput.style.height = "auto";
    updateSendButtonState();
    return;
  }

  // Handle welcome confirmation — starts virtual tour intro, do NOT also forward raw message.
  if (handleWelcomeConfirmation(text, false)) {
    textInput.value = "";
    textInput.style.height = "auto";
    updateSendButtonState();
    return;
  }

  // Send to agent
  sendMessage(text);
  logConversationMessage("user", text);

  // Seek video silently in background based on Qdrant results.
  searchQdrantAndSeekVideo(text);

  textInput.value = "";
  textInput.style.height = "auto";
  updateSendButtonState();
}

if (sendTextButton) {
  sendTextButton.addEventListener("click", submitTextMessage);
}

if (textInput) {
  textInput.addEventListener("input", () => {
    // Auto-resize
    textInput.style.height = "auto";
    textInput.style.height = Math.min(textInput.scrollHeight, 120) + "px";
    updateSendButtonState();
  });

  textInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitTextMessage();
    }
  });
}

// Decode Base64 data to Array
// Handles both standard base64 and base64url encoding
function base64ToArray(base64) {
  // Convert base64url to standard base64
  // Replace URL-safe characters: - with +, _ with /
  let standardBase64 = base64.replace(/-/g, '+').replace(/_/g, '/');

  // Add padding if needed
  while (standardBase64.length % 4) {
    standardBase64 += '=';
  }

  const binaryString = window.atob(standardBase64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes.buffer;
}



/**
 * Audio handling
 */

let audioPlayerNode;
let audioPlayerContext;
let audioRecorderNode;
let audioRecorderContext;
let micStream;
let audioOutputInitPromise = null;
let audioInputInitPromise = null;

// Import the audio worklets
import { startAudioPlayerWorklet } from "./audio-player.js";
import { startAudioRecorderWorklet } from "./audio-recorder.js";

async function ensureAudioOutputStarted() {
  if (audioPlayerNode && audioPlayerContext) {
    if (audioPlayerContext.state === "suspended") {
      await audioPlayerContext.resume();
    }
    return;
  }

  if (!audioOutputInitPromise) {
    audioOutputInitPromise = startAudioPlayerWorklet()
      .then(async ([node, ctx]) => {
        audioPlayerNode = node;
        audioPlayerContext = ctx;
        if (audioPlayerContext.state === "suspended") {
          await audioPlayerContext.resume();
        }
      })
      .catch((error) => {
        audioOutputInitPromise = null;
        throw error;
      });
  }

  await audioOutputInitPromise;
}

async function ensureAudioInputStarted() {
  if (audioRecorderNode && audioRecorderContext && micStream) {
    if (audioRecorderContext.state === "suspended") {
      await audioRecorderContext.resume();
    }
    return;
  }

  if (!audioInputInitPromise) {
    audioInputInitPromise = startAudioRecorderWorklet(audioRecorderHandler)
      .then(async ([node, ctx, stream]) => {
        audioRecorderNode = node;
        audioRecorderContext = ctx;
        micStream = stream;
        if (audioRecorderContext.state === "suspended") {
          await audioRecorderContext.resume();
        }
      })
      .catch((error) => {
        audioInputInitPromise = null;
        throw error;
      });
  }

  await audioInputInitPromise;
}

// Start full audio mode (output + microphone)
function startAudio() {
  ensureAudioOutputStarted().catch((error) => {
    console.error("Failed to start audio output:", error);
    addSystemMessage("Failed to start audio output");
  });
  ensureAudioInputStarted().catch((error) => {
    console.error("Failed to start microphone:", error);
    addSystemMessage("Failed to start microphone");
  });
}

function stopAudioInputCapture() {
  if (audioRecorderNode) {
    try {
      audioRecorderNode.disconnect();
    } catch (error) {
      console.warn("Failed to disconnect audio recorder node:", error);
    }
    audioRecorderNode = null;
  }

  if (micStream) {
    micStream.getTracks().forEach((track) => track.stop());
    micStream = null;
  }

  if (audioRecorderContext) {
    const contextToClose = audioRecorderContext;
    audioRecorderContext = null;
    contextToClose.close().catch((error) => {
      console.warn("Failed to close audio recorder context:", error);
    });
  }

  audioInputInitPromise = null;
}

function setInputMode(mode, silent = false) {
  if (mode !== "text" && mode !== "voice") {
    return;
  }

  inputMode = mode;

  if (textInput) {
    textInput.style.height = "auto";
  }

  if (mode === "voice") {
    startAudio();
    startKeywordRecognizer();
    is_audio = true;

    if (startAudioButton) {
      startAudioButton.classList.add("active");
      startAudioButton.setAttribute("aria-pressed", "true");
      startAudioButton.title = "Voice mode is active";
    }
    if (textInput) {
      textInput.value = "";
      textInput.placeholder = VOICE_TEXT_INPUT_PLACEHOLDER;
      textInput.classList.add("voice-active");
      textInput.disabled = true;
    }

    if (!silent) {
      addConsoleEntry('outgoing', 'Voice Mode Enabled', {
        status: 'Microphone active'
      }, '🎤', 'system');
    }
    return;
  }

  is_audio = false;
  stopKeywordRecognizer();
  stopAudioInputCapture();

  if (startAudioButton) {
    startAudioButton.classList.remove("active");
    startAudioButton.setAttribute("aria-pressed", "false");
    startAudioButton.title = "Toggle full voice mode";
  }
  if (textInput) {
    textInput.disabled = false;
    textInput.placeholder = DEFAULT_TEXT_INPUT_PLACEHOLDER;
    textInput.classList.remove("voice-active");
  }

  if (!silent) {
    addConsoleEntry('outgoing', 'Text Mode Enabled', {
      status: 'Voice capture stopped'
    }, '⌨️', 'system');
  }

  updateSendButtonState();
}

// Voice button — toggles full voice mode
const startAudioButton = document.getElementById("startAudioButton");
if (startAudioButton) {
  startAudioButton.addEventListener("click", () => {
    if (inputMode === "voice") {
      setInputMode("text");
    } else {
      setInputMode("voice");
    }
  });
}
setInputMode("text", true);

if (endConversationButton) {
  endConversationButton.addEventListener("click", endConversation);
}

function closeEndSummaryModal() {
  if (endConversationModal) {
    endConversationModal.classList.remove("show");
  }
}

if (closeEndConversationModal) {
  closeEndConversationModal.addEventListener("click", closeEndSummaryModal);
}

if (closeSummaryButton) {
  closeSummaryButton.addEventListener("click", closeEndSummaryModal);
}

if (endConversationModal) {
  endConversationModal.addEventListener("click", (event) => {
    if (event.target === endConversationModal) {
      closeEndSummaryModal();
    }
  });
}

// Audio recorder handler
function audioRecorderHandler(pcmData) {
  if (websocket && websocket.readyState === WebSocket.OPEN && is_audio) {
    const interruptionUnlocked = Date.now() < interruptionUnlockedUntil;
    const canSend = !isAgentSpeaking || interruptionUnlocked;
    if (!canSend) {
      return;
    }

    // Send audio as binary WebSocket frame (more efficient than base64 JSON)
    websocket.send(pcmData);
    console.log("[CLIENT TO AGENT] Sent audio chunk: %s bytes", pcmData.byteLength);

    // Log to console panel (optional, can be noisy with frequent audio chunks)
    // addConsoleEntry('outgoing', `Audio chunk: ${pcmData.byteLength} bytes`);
  }
}
