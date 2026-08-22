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

    // Stage 1 of the Android port (see docs/guide/android.md): no compiled
    // Kotlin/Java sources yet, only the manifest and the native library.
}

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
