import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
}

// Bytecode for Java 17 so the Android app can use it; builds with any JDK 17+.
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    testImplementation(kotlin("test"))
    testImplementation(libs.sqlite.jdbc)
}

val repoRoot: File = rootDir.parentFile

// The tests read the same kb.sqlite and golden fixtures as the Python pipeline.
val buildKb = tasks.register<Exec>("buildKb") {
    description = "Build kb/build/kb.sqlite with the Python KB compiler."
    workingDir = repoRoot
    commandLine("uv", "run", "python", "kb/build_kb.py")
    inputs.dir(repoRoot.resolve("kb/items"))
    inputs.dir(repoRoot.resolve("kb/cities"))
    inputs.file(repoRoot.resolve("kb/strings.json"))
    inputs.file(repoRoot.resolve("kb/build_kb.py"))
    outputs.file(repoRoot.resolve("kb/build/kb.sqlite"))
}

tasks.test {
    useJUnitPlatform()
    dependsOn(buildKb)
    systemProperty("repo.root", repoRoot.absolutePath)
    inputs.dir(repoRoot.resolve("tests/fixtures/golden"))
    inputs.file(repoRoot.resolve("tests/fixtures/check_cases.json"))
    testLogging {
        events("failed")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
