Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile("c:\Users\Adithya\OneDrive\Desktop\new final\data\audio\test_meeting.wav")
$synth.Speak("Hello team, thank you for joining. First, we need to finalize the quarterly budget. The current budget is set at fifty thousand dollars. Second, the deadline for the major release is next Friday. Let's make sure we hit that deadline.")
$synth.Dispose()
