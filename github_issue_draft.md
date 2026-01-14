# GitHub Issue Draft

**Title:** Feature: Add User Registration to CLI Interaction

**Body:**

### Context
Currently, the `main.py` entry point allows users to log in with an existing email and password, or use a system default. However, there is no interactive flow to **create a new user** directly from the CLI.

### Requirement
Enhance the startup interaction in `main.py` and `user_manager.py` to include a "Register New User" option.

**Proposed Flow:**
1. prompt: "1. Login | 2. Register"
2. If Register:
   - Ask for Email.
   - Ask for Password (and confirm).
   - Create user in Supabase via `UserManager.register(email, password)`.
   - Auto-login with the new credentials.

### Acceptance Criteria
- [ ] User can select "Register" at startup.
- [ ] New user is successfully created in Supabase `users` table.
- [ ] Pipeline proceeds to execution with the new `user_id`.
