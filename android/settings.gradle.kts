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

// core: plain Kotlin/JVM (KB, prompt, verifier, detector post-processing, voting, impact, ...).
// It needs no Android SDK, so it builds and tests anywhere with a JDK.
include(":core")

// app: the Android app. Needs the Android SDK (ANDROID_HOME or local.properties sdk.dir).
include(":app")
