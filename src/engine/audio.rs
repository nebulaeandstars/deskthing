/// An opaque handle to a host-owned sound. Callers can hold one and play it
/// back, but can never look inside it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SoundId(pub(crate) usize);

/// Playback control.
///
/// Deliberately does *not* cover loading: decoding a file is a backend concern
/// and an asynchronous one, so a backend exposes its own loader and hands back
/// a [`SoundId`]. What crosses this boundary is only "play that thing".
///
/// There is no way to ask where playback has reached. macroquad's audio does
/// not expose it, so anything that has to stay in step with a sound has to run
/// off the same wall clock rather than off the sound itself.
pub trait Audio {
    /// Starts a sound, restarting it if already playing.
    fn play(&mut self, sound: SoundId, looped: bool);

    fn stop(&mut self, sound: SoundId);

    /// Silences everything currently playing.
    fn stop_all(&mut self);

    /// Sets a sound's volume, `0.0` to `1.0`.
    fn set_volume(&mut self, sound: SoundId, volume: f32);
}

/// An [`Audio`] that records instead of making noise, for tests.
#[cfg(test)]
#[derive(Debug, Default)]
pub struct RecordingAudio {
    pub calls: Vec<AudioCall>,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AudioCall {
    Play { sound: SoundId, looped: bool },
    Stop(SoundId),
    StopAll,
    SetVolume { sound: SoundId, volume: f32 },
}

#[cfg(test)]
impl RecordingAudio {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether `sound` is playing, following the calls through in order.
    pub fn is_playing(&self, sound: SoundId) -> bool {
        self.calls.iter().fold(false, |playing, call| match call {
            AudioCall::Play { sound: played, .. } if *played == sound => true,
            AudioCall::Stop(stopped) if *stopped == sound => false,
            AudioCall::StopAll => false,
            _ => playing,
        })
    }
}

#[cfg(test)]
impl Audio for RecordingAudio {
    fn play(&mut self, sound: SoundId, looped: bool) {
        self.calls.push(AudioCall::Play { sound, looped });
    }

    fn stop(&mut self, sound: SoundId) {
        self.calls.push(AudioCall::Stop(sound));
    }

    fn stop_all(&mut self) {
        self.calls.push(AudioCall::StopAll);
    }

    fn set_volume(&mut self, sound: SoundId, volume: f32) {
        self.calls.push(AudioCall::SetVolume { sound, volume });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sound_is_not_playing_until_started() {
        let audio = RecordingAudio::new();

        assert!(!audio.is_playing(SoundId(0)));
    }

    #[test]
    fn playing_then_stopping_leaves_it_silent() {
        let mut audio = RecordingAudio::new();

        audio.play(SoundId(0), true);
        assert!(audio.is_playing(SoundId(0)));

        audio.stop(SoundId(0));
        assert!(!audio.is_playing(SoundId(0)));
    }

    #[test]
    fn stop_all_silences_everything() {
        let mut audio = RecordingAudio::new();

        audio.play(SoundId(0), true);
        audio.play(SoundId(1), false);
        audio.stop_all();

        assert!(!audio.is_playing(SoundId(0)));
        assert!(!audio.is_playing(SoundId(1)));
    }

    #[test]
    fn stopping_one_sound_leaves_the_others_alone() {
        let mut audio = RecordingAudio::new();

        audio.play(SoundId(0), true);
        audio.play(SoundId(1), true);
        audio.stop(SoundId(0));

        assert!(!audio.is_playing(SoundId(0)));
        assert!(audio.is_playing(SoundId(1)));
    }

    #[test]
    fn looping_is_recorded() {
        let mut audio = RecordingAudio::new();

        audio.play(SoundId(3), true);

        assert_eq!(
            vec![AudioCall::Play {
                sound: SoundId(3),
                looped: true
            }],
            audio.calls
        );
    }
}
