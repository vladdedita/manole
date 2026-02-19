Feature: Foreground image captioning during directory loading

  Background:
    Given the application is running
    And no directory is currently loaded

  Scenario: Directory with uncached images shows captioning progress
    When the user opens a directory containing 5 uncached images
    Then the LoadingScreen shows "Loading model" as active
    And after model loads, "Indexing files" becomes active
    And after indexing, "Generating summary" becomes active
    And after summary, "Captioning images (0/5)" becomes active
    And the counter updates as each image is captioned
    And after all images are captioned, "Ready" is shown
    And the chat interface becomes available

  Scenario: Directory with no images skips captioning step
    When the user opens a directory containing no images
    Then the LoadingScreen shows "Loading model", "Indexing files", "Generating summary"
    And no "Captioning images" step is shown
    And after summary, "Ready" is shown directly

  Scenario: Directory with all cached images skips captioning step
    When the user opens a directory where all images are already cached
    Then no "Captioning images" step is shown in the LoadingScreen
    And the transition goes from "Generating summary" to "Ready"

  Scenario: Chat is not available during captioning
    When the user opens a directory with uncached images
    And captioning is in progress
    Then the LoadingScreen is displayed
    And the chat input is not accessible

  Scenario: Captioning error does not block ready state
    When the user opens a directory with images
    And all image captioning fails
    Then the app still transitions to "Ready"
    And the chat interface becomes available

  Scenario: Re-index triggers foreground captioning for new images
    Given a directory is already loaded and ready
    When the user clicks "Reindex"
    Then the loading flow restarts
    And only new uncached images are captioned in the foreground
