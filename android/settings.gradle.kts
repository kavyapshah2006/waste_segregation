pluginManagement {
    repositories {
        gradlePluginPortal()
        google()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "waste-segregation"

// Plain Kotlin/JVM: KB lookup, prompt, verifier, off-topic guard. The Android app (WP5) will
// depend on it; it needs no Android SDK, so it builds and tests anywhere with a JDK.
include(":core")
