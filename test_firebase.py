import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(__file__))

from qsignal.firebase_sync import init_firebase, test_firestore_connection

print("=" * 60)
print("  Q-SIGNAL FIREBASE CONNECTIVITY DIAGNOSTIC")
print("=" * 60)

print("\n1. Testing Firebase Admin SDK initialization...")
res = init_firebase()
print("   Init result:", res)

if not res["success"]:
    print(f"\n❌ Initialization failed: {res.get('error')}")
    sys.exit(1)

project_id = res.get("project_id", "traffic-9aac8")
print(f"   Project ID: {project_id}")

print("\n2. Testing Firestore Database read/write access...")
ok, msg = test_firestore_connection()

if ok:
    print(f"\n✅ SUCCESS: {msg}")
    print("Firebase Firestore is properly configured and communicating with Q-Signal!")
else:
    print(f"\n❌ FAILED: {msg}")
    if "Cloud Firestore API has not been used" in msg or "SERVICE_DISABLED" in msg:
        print("\n" + "=" * 60)
        print("  ACTION REQUIRED: Cloud Firestore is not enabled yet")
        print("=" * 60)
        print(f"Your Firebase credentials for project '{project_id}' are valid,")
        print("but Cloud Firestore has not been activated in the project.")
        print("\nTo fix this:")
        print(f"1. Open Firebase Console Firestore:")
        print(f"   https://console.firebase.google.com/project/{project_id}/firestore")
        print("2. Click 'Create database', choose your region and 'Test mode', then click Create.")
        print("   OR enable the API directly in Google Cloud Console:")
        print(f"   https://console.developers.google.com/apis/api/firestore.googleapis.com/overview?project={project_id}")
        print("3. Wait ~1-2 minutes for Google Cloud to activate the database, then rerun this test.")
        print("=" * 60)

