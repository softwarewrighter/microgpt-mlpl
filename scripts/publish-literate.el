;;; publish-literate.el --- Evaluate docs/microgpt.org and export HTML -*- lexical-binding: t; -*-

;; Run by scripts/publish-literate.sh:
;;   emacs -Q --batch -l scripts/publish-literate.el ORG-FILE MLPL-ELISP-DIR MLPL-COMMAND
;; Loads sw-mlpl's Org-babel support, evaluates every block (baking
;; #+RESULTS in), saves the .org, and exports HTML beside it.

(let* ((args command-line-args-left)
       (org-file (expand-file-name (nth 0 args)))
       (elisp-dir (expand-file-name (nth 1 args)))
       (mlpl-command (nth 2 args)))
  (setq command-line-args-left nil)
  (load (expand-file-name "mlpl-all.el" elisp-dir) nil t)
  (require 'ob)
  (require 'org)
  (require 'ox-html)
  (setq org-confirm-babel-evaluate nil
        make-backup-files nil
        enable-local-variables nil      ; the command comes from the caller
        org-html-validation-link nil)
  (org-babel-do-load-languages 'org-babel-load-languages
                               '((mlpl . t) (emacs-lisp . t)))
  (setq org-babel-mlpl-command mlpl-command)
  (with-current-buffer (find-file-noselect org-file)
    (org-babel-mlpl-reset-session)
    (org-babel-remove-result-one-or-many t)
    (org-babel-execute-buffer)
    (save-buffer)
    (org-html-export-to-html)
    (org-babel-mlpl-reset-session)))
