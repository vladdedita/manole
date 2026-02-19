Feature: Manole Installer and First-Launch Setup
  As a non-technical user
  I want to install Manole with a standard platform installer
  And have AI models downloaded automatically on first launch
  So that the app works fully offline after initial setup

  Background:
    Given the user has downloaded the Manole installer for their platform
    And the user has an active internet connection

  # --- Phase 1: Installation ---

  @macos
  Scenario: Standard macOS installation via DMG
    Given the user has downloaded "Manole-1.0.0.dmg"
    When the user opens the DMG
    And drags Manole to the Applications folder
    Then Manole is installed in /Applications
    And the app bundle size is less than 200 MB
    And no AI model files are present in the app bundle yet

  @linux
  Scenario: Standard Linux installation via AppImage
    Given the user has downloaded "Manole-1.0.0.AppImage"
    When the user makes the file executable
    And double-clicks to launch
    Then Manole starts successfully
    And the AppImage size is less than 200 MB

  # --- Phase 2: First Launch Setup ---

  Scenario: First launch shows setup screen
    Given Manole has never been launched before
    When the user launches Manole
    Then a setup screen is displayed instead of the main UI
    And the setup screen shows a welcome message
    And the setup screen lists all required AI models with their sizes
    And model download begins automatically

  Scenario: Model download shows per-file progress
    Given the first-launch setup screen is displayed
    When models are downloading
    Then each model shows its own progress bar
    And each model shows its file size
    And completed models show a checkmark
    And an overall progress indicator is visible

  Scenario: Setup completes and transitions to main UI
    Given all required models have been downloaded
    And all SHA256 checksums have been verified
    Then the setup screen shows "Setup complete!"
    And a "Get Started" button is displayed
    When the user clicks "Get Started"
    Then the main application UI is loaded
    And all AI features are functional

  Scenario: Subsequent launches skip setup
    Given Manole has completed first-launch setup previously
    When the user launches Manole
    Then the main application UI is loaded directly
    And no setup screen is shown
    And no internet connection is required

  # --- Phase 3: Error Handling ---

  Scenario: Network disconnection during download
    Given models are being downloaded
    When the network connection is lost
    Then a message is shown: "Connection lost. Download will resume when connected."
    And a retry button is available
    When the network connection is restored
    Then download resumes from the last downloaded byte
    And no previously downloaded data is re-downloaded

  Scenario: Download resume after app close
    Given models are partially downloaded
    When the user closes the app during download
    And the user relaunches Manole
    Then the setup screen is shown again
    And download resumes from where it left off
    And completed model files are not re-downloaded

  Scenario: Insufficient disk space
    Given the user's disk has less than 2 GB free
    When the first-launch setup begins
    Then a message is shown with the specific space required
    And no download is attempted
    And the message suggests freeing disk space

  Scenario: Corrupted download detected
    Given a model file has been downloaded
    When SHA256 verification fails for that file
    Then the corrupted file is deleted
    And the download for that specific file is retried automatically
    And other completed downloads are not affected

  # --- Phase 4: Offline Operation ---

  Scenario: Fully offline operation after setup
    Given Manole has completed first-launch setup
    And the machine has no internet connection
    When the user launches Manole
    Then the app starts normally
    And all search features work
    And all chat/RAG features work
    And all image captioning features work
    And no error messages about network connectivity are shown
