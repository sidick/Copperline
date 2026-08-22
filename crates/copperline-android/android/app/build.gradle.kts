plugins {
    id("com.android.application") version "9.3.0"
}

android {
    namespace = "org.copperlinehq.copperline"
    compileSdk = 35

    defaultConfig {
        applicationId = "org.copperlinehq.copperline"
        minSdk = 29
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    // The Rust side is built separately (`cargo ndk ... build --release -o
    // app/src/main/jniLibs`, see docs/guide/android.md), not by this Gradle
    // build -- Gradle's default jniLibs.srcDirs (src/main/jniLibs) just
    // packages whatever cargo-ndk already placed there.

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    // The bundled AROS ROM (assets/aros/ at the repo root, the same files
    // src/romsearch.rs finds on desktop) ships as an APK asset, copied in
    // rather than duplicated in this project -- see copyArosAssets below.
    sourceSets {
        getByName("main") {
            // A plain path, not a Provider<Directory>: AGP 9's classic
            // sourceSets DSL rejects providers outright (it can't tell a
            // generated directory from a static one). The Copy task
            // dependency below (dependsOn on the merge-assets tasks) is
            // what actually keeps this populated before packaging.
            assets.srcDir(layout.buildDirectory.dir("generated/assets").get().asFile)
        }
    }

    // Stage 1 of the Android port (see docs/guide/android.md): no compiled
    // Kotlin/Java sources yet, only the manifest and the native library.
}

val copyArosAssets by tasks.registering(Copy::class) {
    from(rootProject.projectDir.resolve("../../../assets/aros")) {
        include("aros-amiga-m68k-rom.bin", "aros-amiga-m68k-ext.bin", "LICENSE")
    }
    into(layout.buildDirectory.dir("generated/assets/aros"))
}

tasks.matching { it.name.startsWith("merge") && it.name.endsWith("Assets") }
    .configureEach { dependsOn(copyArosAssets) }

dependencies {
    // Resolved from google() at build time; this is what supplies the
    // com.google.androidgamesdk.GameActivity class the manifest references --
    // the piece a hand-built APK (no Gradle) cannot get without vendoring
    // the AAR by hand. See docs/guide/android.md.
    implementation("androidx.games:games-activity:4.4.2")
    // GameActivity extends AppCompatActivity but games-activity's POM does
    // not declare it as a dependency; without this, the class fails to
    // verify (superclass unresolved) and the OS reports it as
    // ClassNotFoundException rather than the more specific error.
    implementation("androidx.appcompat:appcompat:1.7.0")
}
