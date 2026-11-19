package wastesegregation.core

import java.util.regex.Pattern

/**
 * Regex flags that behave the same on the desktop JVM (where the core's tests run) and on
 * Android. Java needs Pattern.UNICODE_CHARACTER_CLASS for \w, \s and \b to follow Unicode (so
 * Hindi words count as words); Android's ICU-based regex always uses Unicode classes and throws
 * IllegalArgumentException("UNICODE_CHARACTER_CLASS flag not supported") if the flag is passed.
 */
internal object Regexes {
    val UNICODE_CLASSES: Int = try {
        Pattern.compile("a", Pattern.UNICODE_CHARACTER_CLASS)
        Pattern.UNICODE_CHARACTER_CLASS
    } catch (e: IllegalArgumentException) {
        0
    }
}
