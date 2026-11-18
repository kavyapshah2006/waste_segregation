import java.net.URI
import java.security.MessageDigest

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

val repoRoot: File = rootDir.parentFile

android {
    namespace = "org.wastesegregation.app"
    compileSdk = 37

    defaultConfig {
        applicationId = "org.wastesegregation.app"
        minSdk = 26
        targetSdk = 37
        versionCode = 1
        versionName = "0.1.0"
        // The phone is Arm; the ExecuTorch AAR also ships x86_64, which only an emulator needs.
        ndk { abiFilters += "arm64-v8a" }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Signed with the debug key so the release APK installs with adb; not for a store.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures { compose = true }

    // Model files and the KB are read through file paths: keep them uncompressed in the APK.
    androidResources { noCompress += listOf("pte", "sqlite", "json", "yaml") }
}

/**
 * Small model files go into the APK assets: the KB, the detector (.pte + class names) and the
 * SigLIP 2 text-side JSON. Large ones (Llama 1.1 GB, SigLIP image encoder 94 MB) are pushed to
 * the phone with scripts/push_models.sh. Missing files are skipped, so the app builds before
 * the detector is trained; it then reports the model as missing.
 */
abstract class CollectAppAssets : DefaultTask() {
    @get:InputFiles
    @get:Optional
    abstract val sources: ConfigurableFileCollection

    @get:OutputDirectory
    abstract val outputDir: DirectoryProperty

    @TaskAction
    fun run() {
        val out = outputDir.get().asFile
        out.deleteRecursively()
        out.mkdirs()
        sources.files.filter { it.isFile }.forEach { it.copyTo(File(out, it.name), overwrite = true) }
    }
}

val collectAppAssets = tasks.register<CollectAppAssets>("collectAppAssets") {
    dependsOn(":core:buildKb")
    sources.from(
        repoRoot.resolve("kb/build/kb.sqlite"),
        repoRoot.resolve("models/vision/out/model.pte"),
        repoRoot.resolve("models/vision/out/metadata.yaml"),
        repoRoot.resolve("models/fallback/out/text_embeddings.json"),
        repoRoot.resolve("models/attributes/out/attribute_prompts.json"),
    )
    outputDir.set(layout.buildDirectory.dir("generated/appAssets"))
}

androidComponents {
    onVariants { variant ->
        variant.sources.assets?.addGeneratedSourceDirectory(collectAppAssets, CollectAppAssets::outputDir)
    }
}

/**
 * sherpa-onnx (speech recognition, Apache-2.0) publishes its Android AAR on GitHub releases, not
 * on Maven. Fetched once into app/libs (gitignored) and checked against a pinned SHA-256.
 */
val sherpaVersion = "1.13.8"
val sherpaSha256 = "633c24321e06b1fe79feafa03ea16cbc0f8a286641e2da3559bac91bdb13bd96"
val sherpaAar = file("libs/sherpa-onnx-$sherpaVersion.aar")

fun sha256(f: File): String = MessageDigest.getInstance("SHA-256")
    .digest(f.readBytes()).joinToString("") { "%02x".format(it) }

if (!sherpaAar.isFile) {
    sherpaAar.parentFile.mkdirs()
    val url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/v$sherpaVersion/sherpa-onnx-$sherpaVersion.aar"
    logger.lifecycle("Downloading $url")
    val tmp = File(sherpaAar.path + ".part")
    URI(url).toURL().openStream().use { input -> tmp.outputStream().use { input.copyTo(it) } }
    check(sha256(tmp) == sherpaSha256) { "sherpa-onnx AAR checksum mismatch" }
    tmp.renameTo(sherpaAar)
}

dependencies {
    implementation(project(":core"))
    implementation(libs.executorch.android)
    implementation(files(sherpaAar))
    implementation(libs.camerax.core)
    implementation(libs.camerax.camera2)
    implementation(libs.camerax.lifecycle)
    implementation(libs.camerax.view)
    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.material3)
    implementation(libs.compose.tooling.preview)
    implementation(libs.activity.compose)
    implementation(libs.lifecycle.viewmodel.compose)
    implementation(libs.lifecycle.runtime.compose)
}
